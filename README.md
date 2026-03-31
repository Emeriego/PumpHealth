# WATER PUMP FUNCTIONALITY PREDICTION

This project predicts the functionality status of water pumps across Tanzania using machine learning. Based on features such as location, pump type, installer, water quality, and population served, the models classify pumps as functional, functional needs repair, or non-functional.

 The project includes:

Data cleaning and exploratory analysis to understand pump failure patterns

Feature engineering and model training for multi-class classification

Geospatial visualizations to highlight areas with high failure risk

A dashboard for interactive insights and maintenance planning

The goal is to support decision-making for water infrastructure management and improve access to clean water in Tanzanian communities.

## List of Columns and their description

id – Unique identifier for each waterpoint.

amount_tsh – Total static head (volume of water pumped).

date_recorded – Date when the waterpoint data was recorded.

funder – Organization or individual who funded the pump.

gps_height – Elevation of the pump in meters. that is the altitude.

installer – Organization or individual who installed the pump.

longitude – GPS longitude of the pump location.

latitude – GPS latitude of the pump location.

wpt_name – Name of the waterpoint.

num_private – Number of private water connections. people using it. NB: low number might indicate communal usage like seen in some regions in africa.

basin – River basin where the pump is located.

subvillage – Subvillage of the pump location.

region – Administrative region name.

region_code – Numeric code representing the region.

district_code – Numeric code for the district.

lga – Local government area of the pump.

ward – Administrative ward of the pump.

population – Number of people served by the pump.

public_meeting – Indicates if a public meeting was held for the pump.

recorded_by – Name of the organization/person who recorded the data.

scheme_management – Organization managing the water scheme.

scheme_name – Name of the water scheme.

permit – Whether the pump has a legal permit.

construction_year – Year the pump was constructed.

extraction_type – Method used to extract water.

extraction_type_group – Grouped version of extraction type.

extraction_type_class – Higher-level classification of extraction type.

management – Organization responsible for maintaining the pump.

management_group – Grouped management type.

payment – Payment method for water usage.

payment_type – Grouped payment information.

water_quality – Quality of water (soft, salty, etc.).

quality_group – Grouped water quality categories.

quantity – Amount of water available (enough, seasonal, etc.).

quantity_group – Grouped quantity categories.

source – Water source type (spring, dam, borehole, etc.).

source_type – More specific type of water source.

source_class – Source classification (groundwater, surface, etc.).

waterpoint_type – Physical type of waterpoint (e.g., communal standpipe).

waterpoint_type_group – Grouped waterpoint type.

status_group – Target variable: pump functionality (functional, functional needs repair, non functional).


## HYPOTHESIS.

We propose that:

### Pump type influences functionality

Hypothesis: Some pump types fail more often than others (e.g., handpumps vs communal standpipes).

### Installer affects reliability

Hypothesis: Pumps installed by certain companies/NGOs are more likely to remain functional.

### Geography impacts pump functionality

Hypothesis: Pumps in specific regions, districts, or basins may fail more frequently due to environmental or maintenance challenges.

### Pump age correlates with failure

Hypothesis: Older pumps are more likely to need repairs or be non-functional.

### Population served / usage affects failure

Hypothesis: Pumps serving larger populations (population) might break down more due to higher usage.

### Water quality impacts pump functionality

Hypothesis: hard water is more likely to damage pump faster than soft water.

### Management, that is the organization responsible for maintaining the pump can influence functionality.

Hypothesis: Some organisations are better managers than others.


## RESEARCH QUESTIONS


Do older pumps fail more often than newer ones?

Are pumps serving larger populations more likely to be non-functional?

Are certain regions or basins more prone to pump failures?

How does the installer relate to long-term functionality?

Which pump types have the highest proportion of functional pumps?

Does water quality or quantity influence pump status? 

How does the pump manager relate to long-term functionality?
