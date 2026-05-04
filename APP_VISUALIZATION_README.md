# 📊 Pipeline Dashboard UI & Visualization Guide

This document defines how the application UI should look and how its visualizations should behave, based on the provided layout template and the implemented Streamlit pipeline app.

---

## 🧭 Overview

The application is a **data pipeline control dashboard** designed for:

* Running pipeline stages (scraping → processing → cleaning → export)
* Monitoring dataset progress
* Visualizing corpus and instruction metrics
* Inspecting outputs and intermediate artifacts

The UI should feel like a **modern analytics dashboard**, not a form-based tool.

---

## 🧩 Layout Structure (Based on Template)

The UI follows a **two-panel dashboard layout**:

### 1. Sidebar (Left Panel)

Purpose: Navigation + system identity

**Components:**

* Logo (top)
* Navigation menu (vertical)
* Active section highlight
* Minimal, clean structure

**Navigation Items:**

* Ingest
* Process
* Clean
* Manifest
* Cloud
* Export
* Metrics
* Instruction Metrics

**Behavior:**

* Sticky (always visible)
* Active section clearly highlighted
* No clutter (no deep nesting)

---

### 2. Main Content Area (Right Panel)

Divided into **three vertical sections**:

---

## 🔝 A. Top Bar

**Purpose:** Context + quick status

**Elements:**

* Page title
* Short description (caption)
* Environment indicator (Python interpreter)
* Optional status icon/user indicator

**Style:**

* Minimal height
* Clean spacing
* Subtle separation from content

---

## 📊 B. KPI Summary Row

A row of **4 metric cards**:

* Raw Assets
* Cleaned Documents
* Manifests
* Agent Exports

**Design Rules:**

* Equal width cards
* Large numeric values
* Small contextual delta (e.g., history count)
* Light background contrast

**Behavior:**

* Updates dynamically
* Always visible at top of dashboard

---

## 📈 C. Main Visualization Panels

Two large stacked panels (as shown in template):

---

### Panel 1: **Primary Workspace**

Changes based on selected section.

#### Examples:

**Ingest**

* Left: Controls (buttons, inputs)
* Right: File preview panel

**Process**

* Single action button
* Execution feedback

**Clean**

* 3 action buttons (Scrub, Find Duplicates, Deduplicate)

**Manifest**

* Build button
* JSON preview

**Cloud**

* Sync button
* Status feedback

**Export**

* Export button

---

### Panel 2: **Visualization / Analytics Panel**

Used mainly in:

#### Metrics Section

**Top: KPI Cards**

* Documents
* Tokens
* Vocabulary
* TTR
* Near-dup rate
* Similarity

**Middle: Charts**

* Bar charts for:

  * Lexical metrics
  * Readability
  * Semantic diversity
  * Duplicate rates

**Bottom: Expandable Details**

* Data tables
* JSON payloads
* Section-wise metrics

---

#### Instruction Metrics Section

**Controls:**

* Dataset selectors (Training / Eval / Gold)

**Content:**

* KPI cards
* Distribution charts:

  * Task distribution
  * Language distribution
* Expandable sections for:

  * Cleaning stats
  * Diversity metrics
  * Similarity scores

---

## 📊 Visualization Rules

### Use the Right Component

| Data Type         | Visualization       |
| ----------------- | ------------------- |
| Key numbers       | Metric cards        |
| Comparisons       | Bar charts          |
| Distributions     | Bar charts + tables |
| Raw data          | Dataframes          |
| Debug / full data | Expanders + JSON    |

---

### Chart Guidelines

* Keep charts **simple and readable**
* Avoid overloading with too many metrics
* Always label axes clearly
* Use consistent scaling

---

## 🎨 UI Styling Guidelines

### Color System

* Primary: Teal (`#0F766E`)
* Hover: Dark teal (`#115E59`)
* Background: Light neutral
* Panels: Soft contrast (not pure white)

### Components

* Buttons:

  * Rounded corners
  * Bold text
  * Clear hover state

* Cards:

  * Slight elevation or border
  * Consistent spacing

* Typography:

  * Clear hierarchy:

    * Title
    * Section headers
    * Captions

---

## ⚙️ Interaction Design

### Actions

* Every pipeline step is triggered via a **button**
* Execution feedback appears in a **status block**
* No silent failures

---

### Navigation

* Sidebar controls entire app state
* Each section acts as an **independent workspace**

---

### Responsiveness

* Desktop-first design
* On smaller screens:

  * Columns stack vertically
  * Controls move above outputs

---

## 🚨 Error & Empty States

The UI must handle missing data gracefully:

* Missing files → show **info message**
* JSON parse errors → show **clear error**
* No data → show **empty state message**
* Never break layout

---

## 🧠 UX Principles

The dashboard should feel like:

✔ A **control center**
✔ A **data observability tool**
✔ A **pipeline execution interface**

Not like:

✘ A script runner
✘ A cluttered admin panel

---

## 🔁 Workflow Summary

The user should be able to:

1. Navigate using sidebar
2. Run pipeline steps
3. Instantly see outputs
4. Monitor progress via KPIs
5. Analyze results via charts

---

## 📦 Implementation Reference

This UI is implemented in:

* Streamlit application
* Pipeline scripts integration
* Metrics visualization modules

See implementation details here:


---

## 🚀 Final Experience Goal

The final UI should deliver:

* Clarity → Easy to understand pipeline state
* Speed → Run actions in one click
* Insight → Visualize quality and performance
* Control → Manage full pipeline from one place

---

**Version:** v2.2
**Project:** Sri Lankan Medical Corpus Pipeline

---
