# Water Institute (TWI) Louisiana Energy Analysis Plan (LEAP) Project


## Data Source

Scenario Data Received from The Water Institute

## Model Inputs ("Model Inputs")

## Methodology

### Construction Impacts ("CON Economic Impacts")

#### Initial Data

See sheet "Expenditures" from Invenergy (in Economic Impacts spreadsheet)
Most costs are taken directly from here, with exceptions:
- These adjustments are due to the massively inflated costs of this project over the older project:
	- Installation labor: this is taken from the first Royal Road project and inflated according to GDP (see WP - GDP Deflator in workpapers folder)
	- Other Costs: this is taken from the first Royal Road project and inflated according to GDP (see WP - GDP Deflator in workpapers folder)
- Government spending induced by direct sales tax revenues: Subtract out the model-imputed Direct Parish Taxes for Evangeline, so that only taxes above the model-imputed amount are captured.
- Negative shock from one less farmer: assumes one job per 1200 acres, rounded (in this case, 2 jobs), over the construction timeline (1.75 years)

#### Sector assignment

Sectors are assigned based on the following:
- Non-solar, non-wind electric generation capex: Woodside LNG project, except construction costs allocated to Construction of Power and Communications Structures
- Solar electric generation capex: 2024 Royal Road Invenergy
- Wind: Offshore Wind Buildout Study (2024)
- Electric generation opex: Electric power generation, transmission, and distribution
- Utility Ratepayer impacts: negative shock to labor income (1:1 direct value added and earnings, all other impacts are induced)
- Transportation capex: motor vehicle manufacturing
- Transportation purchase cost: equal to capex; negative shock to labor income
- Fuel cost savings: positive shock to labor income.

#### RPC Assignments

RPCs are assigned using LEIM with the following exceptions:

- LA RPC is halved for LEIM 376 sectors 24 and 374 (non-residential and power/communications structures)
