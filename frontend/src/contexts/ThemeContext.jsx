import { createContext, useContext, useEffect } from 'react';
import { useLocalStorage } from '../hooks/useLocalStorage';

const ThemeContext = createContext();

/**
 * Theme modes
 */
export const THEME_MODES = {
  LIGHT: 'light',
  DARK: 'dark',
  SYSTEM: 'system',
};

/**
 * ThemeProvider component that manages theme state and applies it to the document
 */
export function ThemeProvider({ children }) {
  // Get system preference
  const getSystemTheme = () => {
    if (typeof window !== 'undefined' && window.matchMedia) {
      return window.matchMedia('(prefers-color-scheme: dark)').matches 
        ? THEME_MODES.DARK 
        : THEME_MODES.LIGHT;
    }
    return THEME_MODES.LIGHT;
  };

  // Store user preference in localStorage
  const [themeMode, setThemeMode] = useLocalStorage('theme-mode', THEME_MODES.SYSTEM);
  
  // Compute the actual theme to apply
  const actualTheme = themeMode === THEME_MODES.SYSTEM ? getSystemTheme() : themeMode;

  // Apply theme to document
  useEffect(() => {
    const root = document.documentElement;
    
    // Remove both classes first
    root.classList.remove(THEME_MODES.LIGHT, THEME_MODES.DARK);
    
    // Add the appropriate class
    root.classList.add(actualTheme);
    
    // Update data attribute for CSS
    root.setAttribute('data-theme', actualTheme);
  }, [actualTheme]);

  // Listen for system theme changes
  useEffect(() => {
    if (themeMode !== THEME_MODES.SYSTEM) return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => {
      const newSystemTheme = mediaQuery.matches ? THEME_MODES.DARK : THEME_MODES.LIGHT;
      const root = document.documentElement;
      root.classList.remove(THEME_MODES.LIGHT, THEME_MODES.DARK);
      root.classList.add(newSystemTheme);
      root.setAttribute('data-theme', newSystemTheme);
    };

    // Modern browsers
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    } 
    // Fallback for older browsers
    else if (mediaQuery.addListener) {
      mediaQuery.addListener(handleChange);
      return () => mediaQuery.removeListener(handleChange);
    }
  }, [themeMode]);

  const toggleTheme = () => {
    setThemeMode((prev) => {
      if (prev === THEME_MODES.LIGHT) return THEME_MODES.DARK;
      if (prev === THEME_MODES.DARK) return THEME_MODES.SYSTEM;
      return THEME_MODES.LIGHT;
    });
  };

  const setTheme = (mode) => {
    if (Object.values(THEME_MODES).includes(mode)) {
      setThemeMode(mode);
    }
  };

  const value = {
    themeMode,
    actualTheme,
    setTheme,
    toggleTheme,
    isDark: actualTheme === THEME_MODES.DARK,
    isLight: actualTheme === THEME_MODES.LIGHT,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

/**
 * Custom hook to use the theme context
 */
export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
