## Cargamos paqueterías
# pip install git+https://github.com/milocortes/costs_benefits_ssp.git@main
from costs_benefits_ssp.cb_calculate import CostBenefits
import numpy as np
import pandas as pd 
import sys
import os 

from costs_benefits_ssp.model.cb_data_model import TXTable,CostFactor,TransformationCost,StrategyInteraction

import polars as pl

##---- Definimos directorios
#DIR_PATH = "/home/milo/Documents/egtp/SISEPUEDE/CB/ejecuciones_cb_paquete/lousiana"

DIR_PATH = os.getcwd()

build_path = lambda PATH  : os.path.abspath(os.path.join(*PATH))

### Directorio de salidas de SSP
SSP_RESULTS_PATH = build_path([DIR_PATH,"ssp_salidas"])

### Directorio de configuración de tablas de costos
CB_DEFAULT_DEFINITION_PATH = build_path([DIR_PATH, "cb_factores_costo"])

### Directorio de salidas del módulo de costos y beneficios
OUTPUT_CB_PATH = build_path([DIR_PATH, "cb_resultados"])

### Directorio de salidas del archivo lousiana_plot_data_QA_QC.csv
OUTPUT_LOUSIANA_QA_PATH = build_path([DIR_PATH, "cb_resultados", "qa"])
OUTPUT_LOUSIANA_CB_PATH = build_path([DIR_PATH, "cb_resultados", "cb"])

## Cargamos los datos
ssp_data = pd.read_csv(os.path.join(SSP_RESULTS_PATH, "louisiana.csv"))
att_primary = pd.read_csv(os.path.join(SSP_RESULTS_PATH, "ATTRIBUTE_PRIMARY.csv"))
att_strategy = pd.read_csv(os.path.join(SSP_RESULTS_PATH, "ATTRIBUTE_STRATEGY.csv"))

## Subset ssp data
future_id_base = 0
future_id_compare = int(sys.argv[1])

futures_ids = [future_id_base, future_id_compare]

primary_ids = att_primary[att_primary.future_id.isin(futures_ids)].primary_id.to_list()
#ssp_data = ssp_data[ssp_data.primary_id.isin(primary_ids)]
q = (
    pl.scan_csv(os.path.join(SSP_RESULTS_PATH, "sisepuede_results_sisepuede_run_2025-01-14T17;04;06.975301_WIDE_INPUTS_OUTPUTS.csv"), ignore_errors=True)
    .filter(
        pl.col('primary_id').is_in(primary_ids)
    )
)

ssp_data = q.collect().to_pandas()

primary_id_base = 275275
att_primary.loc[att_primary.primary_id==primary_id_base,"strategy_id"] = 6005


att_strategy_new = pd.DataFrame([(6005, "PFLO: All transformations", "PFLO:ALL_BASE", 0, "All transformations")], columns = ["strategy_id", "strategy", "strategy_code" , "baseline_strategy_id", "description"])
att_strategy = pd.concat([att_strategy, att_strategy_new], ignore_index = True)

all_tx = 'TX:AGRC:DEC_CH4_RICE|TX:AGRC:DEC_EXPORTS|TX:AGRC:DEC_LOSSES_SUPPLY_CHAIN|TX:AGRC:INC_CONSERVATION_AGRICULTURE|TX:AGRC:INC_PRODUCTIVITY|TX:CCSQ:INC_CAPTURE|TX:ENTC:DEC_LOSSES|TX:ENTC:LEAST_COST_SOLUTION|TX:ENTC:TARGET_CLEAN_HYDROGEN|TX:ENTC:TARGET_RENEWABLE_ELEC|TX:FGTV:DEC_LEAKS|TX:FGTV:INC_FLARE|TX:INEN:INC_EFFICIENCY_ENERGY|TX:INEN:INC_EFFICIENCY_PRODUCTION|TX:INEN:SHIFT_FUEL_HEAT|TX:IPPU:DEC_CLINKER|TX:IPPU:DEC_DEMAND|TX:IPPU:DEC_HFCS|TX:IPPU:DEC_N2O|TX:IPPU:DEC_OTHER_FCS|TX:IPPU:DEC_PFCS|TX:LNDU:DEC_DEFORESTATION|TX:LNDU:DEC_SOC_LOSS_PASTURES|TX:LNDU:INC_REFORESTATION|TX:LNDU:INC_SILVOPASTURE|TX:LNDU:PLUR|TX:LSMM:INC_CAPTURE_BIOGAS|TX:LSMM:INC_MANAGEMENT_CATTLE_PIGS|TX:LSMM:INC_MANAGEMENT_OTHER|TX:LSMM:INC_MANAGEMENT_POULTRY|TX:LVST:DEC_ENTERIC_FERMENTATION|TX:LVST:DEC_EXPORTS|TX:LVST:INC_PRODUCTIVITY|TX:PFLO:INC_HEALTHIER_DIETS|TX:PFLO:INC_IND_CCS|TX:SCOE:DEC_DEMAND_HEAT|TX:SCOE:INC_EFFICIENCY_APPLIANCE|TX:SCOE:SHIFT_FUEL_HEAT|TX:SOIL:DEC_LIME_APPLIED|TX:SOIL:DEC_N_APPLIED|TX:TRDE:DEC_DEMAND|TX:TRNS:INC_EFFICIENCY_ELECTRIC|TX:TRNS:INC_EFFICIENCY_NON_ELECTRIC|TX:TRNS:INC_OCCUPANCY_LIGHT_DUTY|TX:TRNS:SHIFT_FUEL_LIGHT_DUTY|TX:TRNS:SHIFT_FUEL_MARITIME|TX:TRNS:SHIFT_FUEL_MEDIUM_DUTY|TX:TRNS:SHIFT_FUEL_RAIL|TX:TRNS:SHIFT_MODE_FREIGHT|TX:TRNS:SHIFT_MODE_PASSENGER|TX:TRNS:SHIFT_MODE_REGIONAL|TX:TRWW:INC_CAPTURE_BIOGAS|TX:TRWW:INC_COMPLIANCE_SEPTIC|TX:WALI:INC_TREATMENT_INDUSTRIAL|TX:WALI:INC_TREATMENT_RURAL|TX:WALI:INC_TREATMENT_URBAN|TX:WASO:DEC_CONSUMER_FOOD_WASTE|TX:WASO:INC_ANAEROBIC_AND_COMPOST|TX:WASO:INC_CAPTURE_BIOGAS|TX:WASO:INC_ENERGY_FROM_BIOGAS|TX:WASO:INC_ENERGY_FROM_INCINERATION|TX:WASO:INC_LANDFILLING|TX:WASO:INC_RECYCLING'

att_strategy['transformation_specification'] = all_tx

# Asumimos que la ejecución con el primary id 75075 es la estrategia PFLO:NET_ZERO
#net_zero = pd.DataFrame([(75075,0,6002,0)], columns=["primary_id", "design_id", "strategy_id", "future_id"])
#att_primary = pd.concat([att_primary, net_zero], ignore_index=True)

strategy_code_base = "PFLO:ALL_BASE"

## Remueve algunas variables de ssp 
#ssp_data = ssp_data.drop(columns = ['yf_agrc_vegetables_and_vines_tonne_ha', 'totalvalue_enfu_fuel_consumed_inen_fuel_furnace_gas'])

## Instanciamos un objeto de la clase CostBenefits 
cb = CostBenefits(ssp_data, att_primary, att_strategy, strategy_code_base)

cb.ssp_data["future_id"] = 0

results_system = cb.compute_system_cost_for_strategy(strategy_code_tx='PFLO:ALL')

results_tx = cb.compute_technical_cost_for_strategy(strategy_code_tx="PFLO:ALL")

# Combina resultados
results_all = pd.concat([results_system, results_tx], ignore_index = True)

#-------------POST PROCESS SIMULATION RESULTS---------------
# Post process interactions among strategies that affect the same variables
results_all_pp = cb.cb_process_interactions(results_all)

# SHIFT any stray costs incurred from 2015 to 2025 to 2025 and 2035
results_all_pp_shifted = cb.cb_shift_costs(results_all_pp)
results_all_pp_shifted["future_id"] = future_id_compare

#now add cost & benefits 
cb_data = results_all_pp_shifted.copy()
#cb_data = cb_data.query("variable!='cb:agrc:crop_value:crops_produced:vegetables'").reset_index(drop = True)
cb_chars = pd.DataFrame([i for i in cb_data.variable.apply(lambda x : x.split(":"))], columns=("name","sector","cb_type","item_1","item_2"))
cb_data = pd.concat([cb_data, cb_chars], axis = 1)

cb_data["value"] /= 1e9 #making all BUSD

# aggregate
cdata = cb_data.groupby(["sector", "cb_type", "strategy_code"]).agg({"value" : "sum"}).reset_index()

#add cb categories  
data_by_cb_type = cdata.groupby(["strategy_code", "cb_type"]).agg({"value" : "sum"})\
                        .reset_index().pivot(index='strategy_code', 
                                            columns='cb_type', 
                                            values='value')\
                        .replace(np.nan, 0.0)\
                        .reset_index()

data_by_cb_type = pd.concat([pd.DataFrame([future_id_compare], columns=["future_id"]), data_by_cb_type], axis = 1)

### Guardamos resultados

OUTPUT_LOUSIANA_QA_FN_PATH = build_path([OUTPUT_LOUSIANA_QA_PATH, f"lousiana_plot_data_QA_QC_{future_id_compare}.csv"])
OUTPUT_LOUSIANA_CB_FN_PATH = build_path([OUTPUT_LOUSIANA_CB_PATH, f"cost_benefit_results_lousiana_{future_id_compare}.csv"])


results_all_pp_shifted.to_csv(OUTPUT_LOUSIANA_CB_FN_PATH, index = False)
data_by_cb_type.to_csv(OUTPUT_LOUSIANA_QA_FN_PATH, index = False)