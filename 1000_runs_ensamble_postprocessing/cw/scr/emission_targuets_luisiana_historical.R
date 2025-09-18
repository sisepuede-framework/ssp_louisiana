# load libraries
library(data.table)
library(stringi)
library(readxl)

rm(list=ls())
options(digits = 10)

# ----------------------------
# Años a procesar (1990–2021)
# ----------------------------
yrs <- as.character(1990:2021)

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

# -----------------------------
# Expandir SOLO IPPU multi-gas
# - divide cada año por #gases
# -----------------------------
df_ippute_multi <- df[Subsector %in% c("ippu","IPPU") & is_multi == TRUE, {
  gases <- split_vec(gas)
  vars  <- split_vec(variable_field)
  ug    <- unique(gases)

  # Variables por gas (por patrón)
  gas_vars_list <- lapply(ug, function(g) {
    pat <- paste0("emission_co2e_", g, "_")
    g_vars <- vars[grepl(pat, vars, fixed = FALSE)]
    if (length(g_vars) == 0L) {
      pat2 <- paste0("_", g, "_")
      g_vars <- vars[grepl(pat2, vars, fixed = FALSE)]
    }
    if (length(g_vars) == 0L) g_vars <- character(0)
    g_vars
  })

  # Construir filas por gas con todas las columnas de años divididas
  out <- lapply(seq_along(ug), function(i) {
    g <- ug[i]
    vals <- as.list(setNames(numeric(length(yrs)), yrs))
    for (yy in yrs) {
      vals[[yy]] <- get(yy) / length(ug)
    }
    c(list(
      Subsector      = Subsector[1],
      gas            = g,
      variable_field = paste(unique(gas_vars_list[[i]]), collapse = ":")
    ), vals)
  })
  rbindlist(out, fill = TRUE)
}, by = .(row_id = .I)]

# --------------------------------------------------------
# Mantener filas NO-multi (incluye IPPU gas simple + otros)
# Conservando todos los años en ancho
# --------------------------------------------------------
keep_cols <- intersect(names(df), c("Subsector","gas","variable_field", yrs))
df_rest <- df[!(Subsector %in% c("ippu","IPPU") & is_multi == TRUE), ..keep_cols]
setnames(df_rest, c("gas"), c("gas"))

# Unificar estructura (asegurar columnas presentes)
for (yy in yrs) if (!yy %in% names(df_rest)) df_rest[, (yy) := NA_real_]

# -----------------------------------
# Combinar y colapsar por Subsector+gas
# - Vars únicas (":"), años sumados
# -----------------------------------
df_out <- rbindlist(list(df_rest, df_ippute_multi), use.names = TRUE, fill = TRUE)

# Colapsar
df_final <- df_out[, {
  # concatenar variables únicas
  v_all <- unique(unlist(strsplit(paste(variable_field, collapse=":"), ":", fixed=TRUE)))
  ans <- list(variable_field = paste(v_all[nzchar(v_all)], collapse = "|"))
  # sumar por cada año
  for (yy in yrs) ans[[yy]] <- sum(get(yy), na.rm = TRUE)
  ans
}, by = .(Subsector, gas)]

# -----------------------------
# Extras (other_fcs, pfcs, hfcs)
# con ceros en todos los años
# -----------------------------
extras <- data.table(
  Subsector = "ippu",
  gas = c("other_fcs", "pfcs", "hfcs"),
  variable_field = c(
    "emission_co2e_other_fcs_ippu_product_use_product_use_ods_other:emission_co2e_other_fcs_ippu_production_chemicals:emission_co2e_other_fcs_ippu_production_electronics",
    "emission_co2e_pfcs_ippu_product_use_product_use_ods_other:emission_co2e_pfcs_ippu_production_chemicals:emission_co2e_pfcs_ippu_production_electronics:emission_co2e_pfcs_ippu_production_other_product_manufacturing",
    "emission_co2e_hfcs_ippu_product_use_product_use_ods_other:emission_co2e_hfcs_ippu_product_use_product_use_ods_refrigeration:emission_co2e_hfcs_ippu_production_chemicals:emission_co2e_hfcs_ippu_production_electronics:emission_co2e_hfcs_ippu_production_metals"
  )
)
for (yy in yrs) extras[, (yy) := 0]

# Unir extras
df_final <- rbindlist(list(df_final, extras), use.names = TRUE, fill = TRUE)

# Homologar nombres antes del merge
setnames(df_final, old = c("gas","variable_field"), new = c("Gas","Vars"))

# -----------------------------
# Cargar emission targets previos
# -----------------------------
df_et <- fread('1000_runs_ensamble_postprocessing/cw/emission_targets_louisiana.csv')
df_et$Gas[df_et$Edgar_Class == "LULUCF - Deforestation:CO2"] <- "co2_def"
df_et <- df_et[, .(Subsector, Gas, Edgar_Class, Edgar_Subsector, Edgar_Sector, ids)]

# Merge con mapping; mantener todos los años
df_merged <- merge(df_et, df_final, by = c("Subsector","Gas"), all = TRUE)

# Mapear co2_def -> co2 (después del merge para respetar mapping)
df_merged$Gas[df_merged$Gas == "co2_def"] <- "co2"

# Limpiar Vars: reemplazar '|' por ':' y quitar duplicados por fila
df_merged[, Vars := gsub("\\|", ":", Vars)]
df_merged[, Vars := sapply(strsplit(Vars, ":"), function(x) paste(unique(x[nzchar(x)]), collapse=":"))]


df_merged$Edgar_Class[is.na(df_merged$Edgar_Class)] <- "IN - Industrial Processes:N2O"
df_merged$Edgar_Subsector[is.na(df_merged$Edgar_Subsector)] <- "IN - Industrial Processes"

# Reordenar columnas clave
final_cols <- c('Subsector','Gas','Edgar_Class','Edgar_Subsector','Edgar_Sector','Vars','ids', yrs)
df_final_out <- df_merged[, ..final_cols]


df_collapsed <- df_final_out[, 
  lapply(.SD, sum, na.rm = TRUE), 
  by = .(Subsector, Edgar_Subsector, Edgar_Class),
  .SDcols = yrs
]

csc_dict <- fread('1000_runs_ensamble_postprocessing/cw/scr/csc_sector_subsector_gas_mapping.csv')


df_classified <- merge(
  df_collapsed,
  csc_dict,
  by = c("Subsector","Edgar_Subsector","Edgar_Class"),
  all.x = TRUE
)

# Vista previa del resultado
head(df_classified)

df_classified$Code <- "LA"

# Escribir a disco
fwrite(df_classified, '1000_runs_ensamble_postprocessing/cw/emission_targets_louisiana_all_years_1990_2021.csv', row.names = FALSE)
