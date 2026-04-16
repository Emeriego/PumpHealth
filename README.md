# WATER PUMP FUNCTIONALITY PREDICTION

This project predicts the functionality status of water pumps across Tanzania using machine learning. Based on features such as location, pump type, installer, water quality, and population served, the models classify pumps as functional, functional needs repair, or non-functional.

 The project includes:

Data cleaning and exploratory analysis to understand pump failure patterns

Feature engineering and model training for multi-class classification

Geospatial visualizations to highlight areas with high failure risk

A dashboard for interactive insights and maintenance planning

The goal is to support decision-making for water infrastructure management and improve access to clean water in Tanzanian communities.

# Water Pump Functionality Prediction (Tanzania)

## Project Overview

This project predicts the functionality status of water pumps across Tanzania using machine learning.

The goal is to support decision-making in water infrastructure maintenance by identifying pumps that are:
- Functional
- Functional but need repair
- Non-functional

By analyzing environmental, geographic, and operational features, the model helps prioritize maintenance efforts and improve access to clean water.

---

## Objectives

- Predict water pump functionality status
- Identify key factors influencing pump failure
- Provide geospatial insights for infrastructure planning
- Support data-driven maintenance strategies

---

## Dataset Description

The dataset contains detailed information about waterpoints in Tanzania.

It includes both **geographical, structural, and operational features** used to predict pump functionality.

### Target Variable
- `status_group` → Pump functionality status:
  - functional
  - functional needs repair
  - non functional

## Feature Dictionary

Below is a structured description of all features used in the dataset:

## Identification & Location
- id – Unique identifier for each waterpoint  
- gps_height – Elevation of the pump in meters  
- longitude – GPS longitude  
- latitude – GPS latitude  
- region – Administrative region  
- region_code – Numeric region identifier  
- district_code – District identifier  
- lga – Local Government Area  
- ward – Administrative ward  
- subvillage – Subvillage location  
- basin – River basin

### Pump Characteristics
- amount_tsh – Total static head (water pressure potential)  
- pump type features:
  - extraction_type
  - extraction_type_group
  - extraction_type_class  
- waterpoint_type – Type of waterpoint  
- waterpoint_type_group – Grouped type  
- construction_year – Year pump was built  
- num_private – Private water connections

### Management & Installation
- installer – Organization that installed pump  
- funder – Organization that funded pump  
- scheme_management – Management authority  
- scheme_name – Name of water scheme  
- management – Responsible maintenance group  
- management_group – Grouped management category  
- recorded_by – Data recording organization  

### Water Usage & Quality
- water_quality – Water quality type  
- quality_group – Grouped quality  
- quantity – Water availability level  
- quantity_group – Grouped quantity  

### Usage & Social Features
- population – Number of people served  
- payment – Payment method  
- payment_type – Grouped payment type  
- public_meeting – Whether public meeting was held  
- permit – Whether pump is legally permitted  

### Source Information
- source – Water source type  
- source_type – Detailed source category  
- source_class – Ground/surface classification  

### Temporal Feature
- date_recorded – Date of data collection
---

## Key Hypotheses

- Pump type affects failure rate
- Older pumps are more likely to fail
- Certain installers produce more reliable pumps
- Geography influences pump functionality
- Higher usage increases breakdown probability
- Water quality impacts pump durability
- Management quality affects long-term performance

---

## Research Questions

- Which factors most influence pump failure?
- Do older pumps fail more often?
- Are certain regions more prone to breakdowns?
- Which installers produce the most reliable pumps?
- How does population usage affect pump status?
- What role does water quality play?
- Which pump types are most sustainable?

---

## Project Workflow

1. Data Cleaning & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training (Multi-class classification)
5. Model Evaluation
6. Geospatial Analysis & Visualization

---

## Project Structure
