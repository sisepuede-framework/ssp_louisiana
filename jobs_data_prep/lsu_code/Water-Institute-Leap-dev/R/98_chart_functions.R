# =============================================================================
# Script Name:     98_graph_functions.R
# Purpose:         R functions for Graphing
# Author:          Ashma Pandey
# Date Created:    
# Last Modified:   2025-07-27 by Garrett Ordner
#
# Inputs:          input
# Outputs:         output
#
# Dependencies:    
# Notes:           
# =============================================================================



# ============================================================================
# 1. COLORS AND PALETTE
# ============================================================================

# CES Color Palette
ces_colors <- list(
  purple = "#662F90",
  gold = "#FBCC34",
  lpurple = "#BD77B2",
  green = "#CBDB2A",
  gray = "#999999",
  lgray = "#D4D4D4",
  black = "#000000",
  red = "#FF0000",
  bluish_black = "#00004D",
  grid_gray = "#E5E5E5",
  subtitle_gray = "#282828"
)

# Get a specific CES color
get_ces_color <- function(name) {
  color <- ces_colors[[name]]
  if (is.null(color)) {
    stop("Color '", name, "' not found in CES palette.")
  }
  color
}

# ============================================================================
# 2. THEME
# ============================================================================

# CES ggplot2 Theme
ces_theme <- function(base_size = 12, base_family = "") {
  ggplot2::theme_minimal(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      # Background and region
      plot.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(color = ces_colors$grid_gray, linewidth = 0.5),
      panel.grid.minor = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),
      
      # Axes and lines
      axis.line = ggplot2::element_line(color = ces_colors$black, linewidth = 0.6),
      axis.ticks = ggplot2::element_line(color = ces_colors$black, linewidth = 0.6),
      axis.ticks.length = ggplot2::unit(.3, "cm"),
      axis.text.x = ggplot2::element_text(color = ces_colors$black, size = base_size * 1.5, 
                                          face = "plain", margin = ggplot2::margin(t = 5)),
      axis.text.y = ggplot2::element_text(color = ces_colors$black, size = base_size * 1.5, 
                                          face = "plain", margin = ggplot2::margin(r = 5)),      
      axis.title.x = ggplot2::element_text(color = ces_colors$black, size = base_size * 1.5,
                                           margin = ggplot2::margin(t = 8, b = 5)),
      axis.title.y.left = ggplot2::element_text(color = ces_colors$black, angle = 90, vjust = 0.5,
                                                size = base_size * 1.5, margin = ggplot2::margin(r = 8)),
      axis.title.y.right = ggplot2::element_text(color = ces_colors$black, angle = 90, vjust = 0.5,
                                                 size = base_size * 1.5, margin = ggplot2::margin(l = 8)),
      
      # Titles and captions
      plot.title = ggplot2::element_text(color = ces_colors$bluish_black, size = base_size * 2,
                                         face = "bold", hjust = 0.5, margin = ggplot2::margin(b = 10)),
      plot.subtitle = ggplot2::element_text(color = ces_colors$subtitle_gray, size = base_size * 1.2,
                                            hjust = 0.5, margin = ggplot2::margin(b = 15)),
      plot.caption = ggplot2::element_text(color = ces_colors$black, size = base_size * 0.75,
                                           hjust = 0, margin = ggplot2::margin(t = 10)),
      
      # Legend
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.title = ggplot2::element_blank(),
      legend.text = ggplot2::element_text(color = ces_colors$black, size = base_size * 1.4),
      legend.background = ggplot2::element_rect(fill = "white", color = "black", linewidth = 0.6),
      legend.key = ggplot2::element_rect(fill = "white", color = NA),
      legend.key.width = ggplot2::unit(30, "mm"),
      legend.key.height = ggplot2::unit(5, "mm"),
      legend.spacing.x = ggplot2::unit(6, "mm"),
      legend.spacing.y = ggplot2::unit(2, "mm"),
      legend.margin = ggplot2::margin(t = 8, r = 8, b = 8, l = 8, unit = "pt"),
      legend.box.margin = ggplot2::margin(t = -4, r = 0, b = 4, l = 0, unit = "pt"),
      
      # Margins
      plot.margin = ggplot2::margin(t = 20, r = 25, b = 30, l = 20, unit = "pt")
    )
}

# ============================================================================
# 3. GEOM WRAPPERS
# ============================================================================

# CES Line Geom
ces_line <- function(mapping = NULL, data = NULL, color = NULL, linewidth = 0.9, linetype = "solid", ...) {
  args <- list(mapping = mapping, data = data, linewidth = linewidth, linetype = linetype, ...)
  if (!is.null(color)) args$color <- color
  do.call(ggplot2::geom_line, args)
}

# CES Point Geom
ces_point <- function(mapping = NULL, data = NULL, color = NULL, size = 2, shape = 16, ...) {
  args <- list(size = size, shape = shape, ...)
  if (!is.null(color)) args$color <- color
  do.call(ggplot2::geom_point, args)
}

# CES Bar/Column Geom
ces_bar <- function(mapping = NULL, data = NULL, fill = NULL, width = 0.8, ...) {
  args <- list(mapping = mapping, data = data, width = width, ...)
  if (!is.null(fill)) args$fill <- fill
  do.call(ggplot2::geom_col, args)
}

# CES Vertical Line
ces_vline <- function(xintercept, color = ces_colors$red, linewidth = 0.2, linetype = "solid", ...) {
  ggplot2::geom_vline(xintercept = xintercept, color = color, linewidth = linewidth, linetype = linetype, ...)
}

# CES Area Geom
ces_area <- function(mapping = NULL, data = NULL, fill = NULL, alpha = 1, ...) {
  args <- list(mapping = mapping, data = data, alpha = alpha, ...)
  if (!is.null(fill)) args$fill <- fill
  do.call(ggplot2::geom_area, args)
}


# ============================================================================
# 4. UTILITY FUNCTIONS
# ============================================================================

# Setup CES Fonts
ces_fonts <- function(font_family = "Source Sans Pro") {
  sysfonts::font_add_google(name = font_family, family = "ces_font")
  showtext::showtext_auto()
  message("Using Google Font: ", font_family)
}

# Add CES Source and Logo
add_ces_source_and_logo <- function(plot,
                                    source_text = "Source: CES Analysis",
                                    logo_path = here::here("R","CESLogo.png"),
                                    source_size = 12) {
  if (!file.exists(logo_path)) {
    message("Logo file not found at: ", logo_path)
    return(plot)
  }
  
  logo_img <- png::readPNG(logo_path)
  logo_grob <- grid::rasterGrob(logo_img, x = 0.99, y = 0.0, width = grid::unit(0.16, "npc"),
                                just = c("right", "bottom"))
  
  cowplot::ggdraw() +
    cowplot::draw_plot(plot) +
    cowplot::draw_label(source_text, x = 0.02, y = 0.02, hjust = 0, vjust = 0, size = source_size,
                        color = get_ces_color("black")) +
    cowplot::draw_grob(logo_grob)
}

# Save CES Plot
save_ces_plot <- function(plot, filename, width = 16, height = 9, units = "in", dpi = 300, ...) {
  ggplot2::ggsave(filename, plot = plot, width = width, height = height, units = units, dpi = dpi, ...)
}
# # ============================================================================
# 5. Chart Builder
make_ces_plot <- function(data, chart_impact_metric, y_column, y_label = NULL, title_label = NULL, output_path = NULL) {
  # Filter and prepare data
  filtered <- data %>%
    filter(
      impact_level == "Total",
      impact_metric == chart_impact_metric
    ) %>%
    arrange(time_period)
  print(filtered %>% select(time_period, all_of(y_column)) %>% distinct())

  y_label <- y_label %||% y_column
  
  # Format y-axis labels
  formatter <- switch(
    chart_impact_metric,
    "Earnings" = scales::label_dollar(scale_cut = cut_short_scale(), accuracy = 1.0),
    "Value Added" = scales::label_dollar(scale_cut = cut_short_scale(), accuracy = 1.0),
    scales::label_comma()
  )
  
  # Create plot
  p <- ggplot(filtered, aes(x = time_period, y = .data[[y_column]])) +
    ces_line(color = get_ces_color("purple")) +
    ces_point(color = get_ces_color("purple")) +
    ces_theme() +
    labs(
      title = paste(title_label, chart_impact_metric, "by Year"),
      x = "Time Period",
      y = NULL
    ) +
    scale_y_continuous(labels = formatter)
  
  final_plot <- add_ces_source_and_logo(p)
  
  if (!is.null(output_path)) {
    save_ces_plot(final_plot, output_path)
  }
  return(final_plot)
}
  