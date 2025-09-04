#load libraries
library(data.table)
library(stringi)
library(readxl)

rm(list=ls())

options(digits = 10)

# historical emissions
df <- fread('1000_runs_ensamble_postprocessing/cw/scr/Louisiana_historical_emissions.csv')
setDT(df)

# --- Helper: robust splitter for c("a","b") OR "a|b" OR "a,b" ---
split_vec <- function(x) {
  if (is.list(x)) return(x)
  x <- as.character(x)
  x <- stri_trim_both(x)
  x <- sub("^c\\s*\\(", "", x, perl = TRUE)
  x <- sub("\\)\\s*$", "", x, perl = TRUE)
  x <- gsub('(^"|"$)', "", x)
  x <- gsub('"', "", x, fixed = TRUE)
  x <- gsub("'", "", x, fixed = TRUE)
  parts <-
    if (grepl("\\|", x)) strsplit(x, "\\|", perl = TRUE)[[1]]
  else                 strsplit(x, "\\s*,\\s*", perl = TRUE)[[1]]
  parts <- stri_trim_both(parts)
  parts[nzchar(parts)]
}

# Detect rows that are "multi-gas" (anidados en una sola fila)
df[, is_multi := grepl("\\|", gas) | grepl("^c\\s*\\(", gas)]

# --- Expand ONLY the multi-gas IPPU rows, splitting total 2021 evenly by gas ---
df_ippute_multi <- df[Subsector %in% c("ippu","IPPU") & is_multi == TRUE, {
  gases <- split_vec(gas)
  vars  <- split_vec(variable_field)
  
  # gases únicos (por si algún gas viene repetido)
  ug    <- unique(gases)
  per_gas_val <- `2021` / length(ug)
  
  # Para cada gas, toma TODAS las variables que contengan ese gas en el nombre
  # (p.ej. "emission_co2e_c2f6_..."), y concaténalas con ":"
  out <- lapply(ug, function(g) {
    pat <- paste0("emission_co2e_", g, "_")
    g_vars <- vars[grepl(pat, vars, fixed = FALSE)]
    # Si no encontró por patrón, como fallback intenta por "_g_"
    if (length(g_vars) == 0L) {
      pat2 <- paste0("_", g, "_")
      g_vars <- vars[grepl(pat2, vars, fixed = FALSE)]
    }
    list(
      Subsector      = Subsector[1],
      gas            = g,
      variable_field = paste(unique(g_vars), collapse = ":"),
      value_2021     = per_gas_val
    )
  })
  rbindlist(out, fill = TRUE)
}, by = .(row_id = .I)]

# --- Mantén tal cual las filas NO-multi (incluye IPPU con gas simple y otros subsectores) ---
df_rest <- df[!(Subsector %in% c("ippu","IPPU") & is_multi == TRUE),
              .(Subsector, gas, variable_field, value_2021 = `2021`)]

# --- Combina resultado ---
df_out <- rbindlist(list(df_rest, df_ippute_multi), use.names = TRUE, fill = TRUE)

# (Opcional) Si quieres colapsar a un registro por Subsector+gas:
#   - Concatenar variables únicas con ":"
#   - Sumar valores 2021
df_final <- df_out[, .(
  variable_field = paste(unique(unlist(strsplit(variable_field, ":", fixed = TRUE))), collapse = "|"),
  value_2021     = sum(value_2021, na.rm = TRUE)
), by = .(Subsector, gas)]

# Definir gases extra como data.table
extras <- data.table(
  Subsector = "ippu",  # ajusta si quieres otro valor en Subsector
  gas = c("other_fcs", "pfcs", "hfcs"),
  variable_field = c(
    "emission_co2e_other_fcs_ippu_product_use_product_use_ods_other:emission_co2e_other_fcs_ippu_production_chemicals:emission_co2e_other_fcs_ippu_production_electronics",
    "emission_co2e_pfcs_ippu_product_use_product_use_ods_other:emission_co2e_pfcs_ippu_production_chemicals:emission_co2e_pfcs_ippu_production_electronics:emission_co2e_pfcs_ippu_production_other_product_manufacturing",
    "emission_co2e_hfcs_ippu_product_use_product_use_ods_other:emission_co2e_hfcs_ippu_product_use_product_use_ods_refrigeration:emission_co2e_hfcs_ippu_production_chemicals:emission_co2e_hfcs_ippu_production_electronics:emission_co2e_hfcs_ippu_production_metals"
  ),
  value_2021 = 0
)

# Unirlos al resultado final
df_final <- rbindlist(list(df_final, extras), use.names = TRUE, fill = TRUE)
colnames(df_final) <- c("Subsector", "Gas", "Vars", "LA")

# past emissions targets
df_et <- fread('1000_runs_ensamble_postprocessing/cw/emission_targets_louisiana.csv')
df_et$Gas[df_et$Edgar_Class== "LULUCF - Deforestation:CO2"] <- "co2_def"

# merge to old targets 
df_final <- merge(df_et, df_final, by = c("Subsector", "Gas"), all = TRUE)

df_final <- df_final[, .(Subsector, Gas, Edgar_Class, Edgar_Subsector, Edgar_Sector, Vars.y, ids, LA.y)]
colnames(df_final) <- c('Subsector', 'Gas', 'Edgar_Class', 'Edgar_Subsector', 'Edgar_Sector', 'Vars', 'ids', 'LA')


df_final$Vars <- gsub("\\|", ":", df_final$Vars)
df_final$Gas[df_final$Gas== "co2_def"] <- "co2"


# Remove duplicate Vars in each row
df_final$Vars <- sapply(strsplit(df_final$Vars, ":"), function(x) paste(unique(x), collapse = ":"))

# Resultado listo:
head(df_final)


fwrite(df_final,'1000_runs_ensamble_postprocessing/cw/emission_targets_louisiana_250903.csv')

