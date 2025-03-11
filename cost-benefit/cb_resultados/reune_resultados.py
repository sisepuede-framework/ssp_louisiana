import pandas as pd 
import glob


df_cb = pd.concat([pd.read_csv(i) for i in glob.glob("cb/*.csv")], ignore_index = False)
df_cb.to_csv("cost_benefit_results_lousiana.csv", index = False)

df_qa = pd.concat([pd.read_csv(i) for i in glob.glob("qa/*.csv")], ignore_index = False)
df_qa.to_csv("lousiana_plot_data_QA_QC.csv", index = False)