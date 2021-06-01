library(MSnbase)
library(stringr)
library(xlsx)
library(optparse)
# Create an intensity matrix from a list of MS spectra
# spectra_files must be in mzXML format
import_sp <- function(spectra_files) {
  get_labels <- function (sp_data) {

    assays_names <- ls(assayData(sp_data))
    labels <- str_split(spectra_files, '/')

    i <- 1
    for (label in labels) {
      # TODO look for labels in sp_data instead
      # label <- label[[1]]
      labels[[i]] <- str_split(str_split(label[length(label)], '_')[[1]][3], '-')[[1]][1]
      i <- i + 1
    }


    i <- 1
    for (assay_name in assays_names) {
      sample_num <- str_split(str_split(assay_name, '.S')[[1]][1], 'F')[[1]][2]
      sample_num <- strtoi(gsub("(?<![0-9])0+", "", sample_num, perl = TRUE))
      assays_names[[i]] <- labels[sample_num]
      i <- i + 1
    }
    return(assays_names)
  }

  # Use spectra only with MS level 1
  sp_data <- readMSData(spectra_files, msLevel=1)
  # TIC (total ion count) > 1e4
  sp_data <- sp_data[tic(sp_data)> 1e4]
  labels <- get_labels(sp_data)
  labels <- na.omit(labels)
  for (i in 1:length(labels)) {
    labels[[i]] <- labels[[i]][[1]]
  }

  # Bin the intensities values according to the bin size
  bined_sp <- do.call(rbind, MSnbase::intensity(bin(sp_data, binSize=0.1)))

  return(list(bined_sp, labels))
}

option_list <- list(
  make_option("--outIntensities", type="character", default="data/beef_intensities.csv",
              help="Output files for intensities", metavar="character"),
  make_option("--dir", type="character", default="D:\\workbench\\data\\spectro\\[SpiderMass]-Databank_Synapt_Beef-liver_Reproducibility\\20170228_Bl_Reproducibility\\copy_mzXML",
              help="spectro directory name", metavar="character"),
  make_option("--meta", type="character", default="D:\\workbench\\data\\spectro\\[SpiderMass]-Databank_Synapt_Beef-liver_Reproducibility\\20170228_Bl_Reproducibility\\20170209_Bl_Reproducibility-experiment_Sample-list.xlsx",
              help="spectro directory name", metavar="character"),
  make_option("--label_column", type="character", default="Analysis.time",
              help="label column", metavar="character")
);

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);

spectra_files <- list.files(opt$dir, full.names = TRUE)
tmp <- import_sp(spectra_files)
intensity_matrix <- tmp[[1]]
labels <- tmp[[2]]

file_sp <- file(opt$outIntensities, open="wt")
metadata <- read.xlsx(opt$meta, 1)
labels_names <- metadata[opt$label_column][[1]]
labels_names <- na.omit(labels_names)

for (i in 1:length(labels)) {
  labels[[i]] <- paste(labels_names[strtoi(labels[[i]][[1]])], strtoi(labels[[i]][[1]]), sep='_')
}
# write.table(intensity_matrix, file_sp, sep=",", row.names=FALSE)
# write.csv(labels, file =opt$outLabels, row.names=FALSE)
rownames(intensity_matrix) <- labels
write.table(intensity_matrix, file_sp, sep=",", row.names=TRUE)
