import { useState, useEffect } from 'react';

const BASE_DELAY = 1000; // 1 second
const MAX_RETRIES = 3;
const BACKOFF_MULTIPLIER = 2;
const API_BASE = (
  globalThis.__APP_CONFIG__?.VITE_API_BASE ||
  import.meta.env.VITE_API_BASE ||
  ''
).replace(/\/$/, '');
const IS_PRODUCTION = Boolean(import.meta.env.PROD);

/**
 * Enhanced API service with retry logic and better error handling
 */
class ApiService {
  constructor(baseURL = API_BASE) {
    this.baseURL = baseURL;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  /**
   * Sleep utility for retry delays
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Calculate exponential backoff delay
   */
  getBackoffDelay(attempt) {
    return BASE_DELAY * Math.pow(BACKOFF_MULTIPLIER, attempt);
  }

  /**
   * Generic request method with retry logic
   */
  async request(endpoint, options = {}, retries = MAX_RETRIES) {
    if (!this.baseURL && IS_PRODUCTION) {
      throw new ApiError(
        'Frontend API is not configured. Set VITE_API_BASE to the backend Railway URL and redeploy the frontend.',
        0,
        { endpoint }
      );
    }

    const url = `${this.baseURL}${endpoint}`;
    const config = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    };

    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const response = await fetch(url, config);
        const contentType = response.headers.get('content-type') || '';
        
        // Handle HTTP errors
        if (!response.ok) {
          const errorData = contentType.includes('application/json')
            ? await response.json().catch(() => ({}))
            : { message: `HTTP ${response.status}: ${response.statusText}` };
          throw new ApiError(
            errorData.message || `HTTP ${response.status}: ${response.statusText}`,
            response.status,
            errorData
          );
        }

        if (!contentType.includes('application/json')) {
          const body = await response.text();
          const looksLikeHtml = /^\s*<!doctype html>|^\s*<html/i.test(body);
          const message = looksLikeHtml
            ? 'Received HTML instead of API JSON. VITE_API_BASE likely points to the frontend or is missing; set it to the backend Railway URL and redeploy.'
            : 'API returned an unexpected non-JSON response.';

          throw new ApiError(message, response.status || 0, {
            endpoint,
            url,
            contentType,
          });
        }

        return await response.json();
      } catch (error) {
        // Don't retry on client errors (4xx) or last attempt
        if (error instanceof ApiError && (error.status >= 400 && error.status < 500 || error.status === 0)) {
          throw error;
        }

        if (attempt === retries) {
          if (error instanceof ApiError) {
            throw error;
          }
          throw new ApiError(
            error.message || 'Network request failed',
            0,
            { originalError: error }
          );
        }

        // Wait before retrying with exponential backoff
        const delay = this.getBackoffDelay(attempt);
        console.warn(`Request failed (attempt ${attempt + 1}/${retries + 1}), retrying in ${delay}ms...`);
        await this.sleep(delay);
      }
    }
  }

  /**
   * GET request
   */
  async get(endpoint, options = {}) {
    return this.request(endpoint, { ...options, method: 'GET' });
  }

  /**
   * POST request
   */
  async post(endpoint, data, options = {}) {
    return this.request(endpoint, {
      ...options,
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /**
   * DELETE request
   */
  async delete(endpoint, options = {}) {
    return this.request(endpoint, { ...options, method: 'DELETE' });
  }

  /**
   * Health check endpoint
   */
  async healthCheck() {
    return this.get('/api/health');
  }

  /**
   * Send chat message
   */
  async sendChatMessage(message, sessionId = null) {
    return this.post('/api/chat', {
      message,
      session_id: sessionId,
    });
  }

  /**
   * Reset session
   */
  async resetSession(sessionId) {
    return this.delete(`/api/session/${sessionId}`);
  }
}

/**
 * Custom error class for API errors
 */
export class ApiError extends Error {
  constructor(message, status, data = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }

  isNetworkError() {
    return this.status === 0;
  }

  isServerError() {
    return this.status >= 500;
  }

  isClientError() {
    return this.status >= 400 && this.status < 500;
  }
}

// Create singleton instance
const api = new ApiService();

// Export named functions for backward compatibility
export const healthCheck = () => api.healthCheck();
export const sendChatMessage = (message, sessionId) => api.sendChatMessage(message, sessionId);
export const resetSession = (sessionId) => api.resetSession(sessionId);

// Export the service instance as default
export default api;
