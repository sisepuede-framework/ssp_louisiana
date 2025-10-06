## 3. Color palette and utilities
# File: R/ces_colors.R

#' CES Color Palette
#'
#' A collection of colors following CES branding guidelines
#'
#' @return A named list of color values
#' @export
#' @examples
#' ces_colors
ces_colors <- list(
  purple = "#662F90",
  gold = "#FBCC34",
  lpurple = "#BD77B2",
  green = "#CBDB2A",
  gray = "#999999",
  lgray = "#D4D4D4",
  black = "#000000",
  red = "#FF0000",
  bluish_black = "#00004D"
)

#' CES Color Palette as Vector
#'
#' @return A named vector of color values
#' @export
ces_palette <- unlist(ces_colors)

#' Get CES Color by Name
#'
#' Retrieve a specific color from the CES palette
#'
#' @param name Name of the color to retrieve
#' @return A color value (hex code)
#' @export
#' @examples
#' get_ces_color("purple")
#' get_ces_color("gold")
get_ces_color <- function(name) {
  if (!name %in% names(ces_colors)) {
    stop(paste0("Color '", name, "' not found in CES palette."))
  }
  ces_colors[[name]]
}

#' CES Color Scale (Discrete)
#'
#' A discrete color scale using CES colors
#'
#' @param discrete Logical, whether to use discrete scale
#' @param reverse Logical, whether to reverse the palette
#' @param ... Additional arguments passed to scale functions
#' @return A ggplot2 scale
#' @export
#' @examples
#' library(ggplot2)
#' ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
#'   geom_point() +
#'   scale_color_ces()
scale_color_ces <- function(discrete = TRUE, reverse = FALSE, ...) {
  values <- if (reverse) rev(ces_palette) else ces_palette
  if (discrete) {
    ggplot2::scale_color_manual(values = values, ...)
  } else {
    ggplot2::scale_color_gradient(low = ces_colors$purple, high = ces_colors$gold, ...)
  }
}

#' CES Fill Scale (Discrete)
#'
#' A discrete fill scale using CES colors
#'
#' @param discrete Logical, whether to use discrete scale
#' @param reverse Logical, whether to reverse the palette
#' @param ... Additional arguments passed to scale functions
#' @return A ggplot2 scale
#' @export
scale_fill_ces <- function(discrete = TRUE, reverse = FALSE, ...) {
  values <- if (reverse) rev(ces_palette) else ces_palette
  if (discrete) {
    ggplot2::scale_fill_manual(values = values, ...)
  } else {
    ggplot2::scale_fill_gradient(low = ces_colors$purple, high = ces_colors$gold, ...)
  }
}

#' Show CES Color Palette
#'
#' Display the CES color palette visually
#'
#' @return A ggplot2 object showing the color palette
#' @export
#' @examples
#' show_ces_palette()
show_ces_palette <- function() {
  df <- data.frame(
    color = names(ces_colors),
    value = 1:length(ces_colors),
    hex = unlist(ces_colors)
  )
  p <- ggplot2::ggplot(df, ggplot2::aes(x = value, y = 1, fill = color)) +
    ggplot2::geom_col(width = 0.8) +
    ggplot2::scale_fill_manual(values = ces_colors) +
    ggplot2::geom_text(ggplot2::aes(label = paste(color, "\n", hex)),
                       color = "black", fontface = "bold", size = 3) +
    ggplot2::labs(title = "CES Color Palette", subtitle = "Official colors for CES graphics") +
    ggplot2::theme_void() +
    ggplot2::theme(
      legend.position = "none",
      plot.title = ggplot2::element_text(hjust = 0.5, size = 15, face = "bold"),
      plot.subtitle = ggplot2::element_text(hjust = 0.5, size = 11)
    )
  return(p)
}

## 4. Main theme function
# File: R/ces_theme.R

#' CES ggplot2 Theme
#'
#' A custom ggplot2 theme following CES style guidelines
#'
#' @param base_size Base font size in points
#' @param base_family Base font family
#' @return A ggplot2 theme object
#' @export
#' @examples
#' library(ggplot2)
#' ggplot(mtcars, aes(x = wt, y = mpg)) +
#'   geom_point() +
#'   ces_theme()
ces_theme <- function(base_size = 12, base_family = "") {
  ggplot2::theme_minimal(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      # Background and region
      plot.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(color = "#E5E5E5", linewidth = 0.5),
      panel.grid.minor = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),

      # Axes and lines
      axis.line = ggplot2::element_line(color = ces_colors$black, linewidth = 0.5),
      axis.ticks = ggplot2::element_line(color = ces_colors$black, linewidth = 0.5),
      axis.text = ggplot2::element_text(color = ces_colors$black, size = base_size * 1.4, face = "plain"),
      axis.title.x = ggplot2::element_text(color = ces_colors$black, size = base_size * 1.4,
                                           margin = ggplot2::margin(t = 8, b = 5)),
      axis.title.y.left = ggplot2::element_text(color = ces_colors$black, angle = 90, vjust = 0.5,
                                                size = base_size * 1.4, margin = ggplot2::margin(r = 8)),
      axis.title.y.right = ggplot2::element_text(color = ces_colors$black, angle = 90, vjust = 0.5,
                                                 size = base_size * 1.4, margin = ggplot2::margin(l = 8)),

      # Titles and captions
      plot.title = ggplot2::element_text(color = ces_colors$bluish_black, size = base_size * 2,
                                         face = "bold", hjust = 0.5, margin = ggplot2::margin(b = 10)),
      plot.subtitle = ggplot2::element_text(color = "#282828", size = base_size * 1.2,
                                            hjust = 0.5, margin = ggplot2::margin(b = 15)),
      plot.caption = ggplot2::element_text(color = ces_colors$black, size = base_size * 0.75,
                                           hjust = 0, margin = ggplot2::margin(t = 10)),

      # Facets
      strip.background = ggplot2::element_rect(fill = ces_colors$lgray, color = NA),
      strip.text = ggplot2::element_text(color = ces_colors$black, size = base_size * 0.8, face = "bold"),

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
      legend.box.margin = ggplot2::margin(t = -8, r = 0, b = 0, l = 0, unit = "pt"),

      # Margins
      plot.margin = ggplot2::margin(t = 20, r = 25, b = 30, l = 20, unit = "pt")
    )
}

## 5. Geom wrapper functions
# File: R/ces_geoms.R

#' CES Line Geom
#'
#' Wrapper for geom_line with CES defaults
#'
#' @param mapping Aesthetic mappings
#' @param data Data frame
#' @param color Line color (optional)
#' @param linewidth Line width
#' @param linetype Line type
#' @param ... Additional arguments passed to geom_line
#' @return A ggplot2 layer
#' @export
#' @examples
#' library(ggplot2)
#' ggplot(economics, aes(x = date, y = unemploy)) +
#'   ces_line(aes(y = unemploy))
ces_line <- function(mapping = NULL, data = NULL, color = NULL, linewidth = 0.9, linetype = "solid", ...) {
  if (is.null(color)) {
    ggplot2::geom_line(mapping = mapping, data = data, linewidth = linewidth, linetype = linetype, ...)
  } else {
    ggplot2::geom_line(mapping = mapping, data = data, color = color, linewidth = linewidth, linetype = linetype, ...)
  }
}

#' CES Point Geom
#'
#' Wrapper for geom_point with CES defaults
#'
#' @param mapping Aesthetic mappings
#' @param data Data frame
#' @param color Point color (optional)
#' @param size Point size
#' @param shape Point shape
#' @param ... Additional arguments passed to geom_point
#' @return A ggplot2 layer
#' @export
ces_point <- function(mapping = NULL, data = NULL, color = NULL, size = 2, shape = 16, ...) {
  if (is.null(color)) {
    ggplot2::geom_point(mapping = mapping, data = data, size = size, shape = shape, ...)
  } else {
    ggplot2::geom_point(mapping = mapping, data = data, color = color, size = size, shape = shape, ...)
  }
}

#' CES Bar/Column Geom
#'
#' Wrapper for geom_col with CES defaults
#'
#' @param mapping Aesthetic mappings
#'  You must explicitly map x (bar positions) and y (bar heights) 
#'  to columns in \code{data}. This function does not infer columns automatically.
#'  Example: mapping = aes(x = category, y = total)
#' @param data Data frame
#' @param fill Fill color (optional)
#' @param width Bar width
#' @param ... Additional arguments passed to geom_col
#' @return A ggplot2 layer
#' @export
ces_bar <- function(mapping = NULL, data = NULL, fill = NULL, width = 0.8, ...) {
  if (is.null(fill)) {
    ggplot2::geom_col(mapping = mapping, data = data, width = width, ...)
  } else {
    ggplot2::geom_col(mapping = mapping, data = data, fill = fill, width = width, ...)
  }
}

#' CES Vertical Line
#'
#' Wrapper for geom_vline with CES defaults
#'
#' @param xintercept X-axis intercept value
#' @param color Line color
#' @param linewidth Line width
#' @param linetype Line type
#' @param ... Additional arguments passed to geom_vline
#' @return A ggplot2 layer
#' @export
ces_vline <- function(xintercept, color = ces_colors$red, linewidth = 0.2, linetype = "solid", ...) {
  ggplot2::geom_vline(xintercept = xintercept, color = color, linewidth = linewidth, linetype = linetype, ...)
}

## 6. Utility functions
# File: R/ces_utils.R

#' Add CES Source and Logo
#'
#' Add source text and CES logo to a plot
#'
#' @param plot A ggplot2 object
#' @param source_text Source text to display
#' @param logo_path Path to CES logo file
#' @return A plot with source and logo added
#' @export
#' @examples
#' p <- ggplot(mtcars, aes(x = wt, y = mpg)) + geom_point()
#' add_ces_source_and_logo(p, "Source: Motor Trend")
add_ces_source_and_logo <- function(plot,
                                    source_text = "Source: CES Analysis",
                                    logo_path = "~/Library/CloudStorage/Box-Box/CES-Research-Share/Graphic Function R/CESLogo.png") {
  if (!file.exists(logo_path)) return(plot)

  logo_img <- png::readPNG(logo_path)
  logo_grob <- grid::rasterGrob(logo_img, x = 0.99, y = 0.0, width = grid::unit(0.2, "npc"),
                                just = c("right", "bottom"))

  cowplot::ggdraw() +
    cowplot::draw_plot(plot) +
    cowplot::draw_label(source_text, x = 0.02, y = 0.02, hjust = 0, vjust = 0, size = 12,
                        color = get_ces_color("black")) +
    cowplot::draw_grob(logo_grob)
}

#' Setup CES Fonts
#'
#' Load and setup Google Fonts for CES graphics
#'
#' @param font_family Font family name from Google Fonts
#' @export
#' @examples
#' ces_fonts("Source Sans Pro")
ces_fonts <- function(font_family = "Source Sans Pro") {
  if (!requireNamespace("sysfonts", quietly = TRUE)) {
    stop("Package 'sysfonts' needed for this function to work. Please install it.")
  }
  if (!requireNamespace("showtext", quietly = TRUE)) {
    stop("Package 'showtext' needed for this function to work. Please install it.")
  }

  sysfonts::font_add_google(name = font_family, family = "ces_font")
  showtext::showtext_auto()
  message("Using Google Font: ", font_family)
}

#' Save CES Plot
#'
#' Save a plot with CES-optimized settings
#'
#' @param plot ggplot2 object
#' @param filename Output filename
#' @param width Plot width
#' @param height Plot height
#' @param units Units for width and height
#' @param ... Additional arguments passed to ggsave
#' @export
#' @examples
#' p <- ggplot(mtcars, aes(x = wt, y = mpg)) + geom_point()
#' save_ces_plot(p, "my_plot.png")
save_ces_plot <- function(plot, filename, width = 1500, height = 850, units = "px", ...) {
  ggplot2::ggsave(filename, plot = plot, width = width, height = height, units = units, dpi = 100, ...)
}

## 7. Package documentation
# File: R/CESgraphics-package.R

#' CESgraphics: CES Graphics Theme and Utilities
#'
#' A collection of ggplot2 themes, color palettes, and utility functions
#' for creating consistent, professional graphics following CES style guidelines.
#'
#' @docType package
#' @name CESgraphics
#' @import ggplot2
#' @import png
#' @import scales
#' @import cowplot
#' @import magick
#' @import sysfonts
#' @import showtext
NULL

## 8. Package loading function
# File: R/zzz.R

.onLoad <- function(libname, pkgname) {
  # Install required packages if not available
  required_packages <- c("ggplot2", "grid", "png", "scales", "cowplot", "magick", "sysfonts", "showtext")
  for(pkg in required_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      message(paste("Package", pkg, "is required but not installed. Please install it."))
    }
  }
}
