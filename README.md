The code that supports the findings in our paper titled “Cumulative Learning Enables Convolutional Neural Network Representations for Small Mass Spectrometry Data Classification”

Describe how to use pwiz.ProteowizardWrapper.MSDataFileReader

**Data availability**

All MS data can be found in : Canine sarcoma raw library is accessible on the ProteomeXchange consortium: PXD010990. Human ovarian datasets can be accessed through FDA-NCI Clinical Proteomics at https://home.ccr.cancer.gov/ncifdaproteomics/ppatterns.asp. Microorganisms, beef liver, and rat brain raw libraries are accessible on https://data.mendeley.com/datasets/33cbb37cs2/1.

**Data description**

Raw SpiderMass spectra are converted into mzXML format using the 64-bit MSConvert tool (version 3.0), part of the ProteoWizard suite. Spectra with a total ion count (TIC) exceeding 1e4 count for irradiation detection are selected using the MSnbase package (version 1.20.7, R version 3.4.4) and converted into a csv file. Raw ovarian datasets are imported into a csv file format.

**Citation request**

<a id="1">[1]</a> 
Please consider citing the following paper:
Seddiki K., Saudemont K., Precioso F., Ogrinc N., Wisztorski M., Salzet M., Fournier I., Arnaud Droit A. submitted. Cumulative Learning with Convolutional Neural Networks Enables Small Mass Spectrometry Data Classification. 

**Contributing**

For any questions, feel free to open an issue or contact at arnaud.droit@crchudequebec.ulaval.ca

**Additional informations**
Might need to rename the files to automate everything. 

R command line example
"C:\Program Files\R\R-4.0.5\bin\Rscript.exe" src\preprocess\import_data.R --outIntensities data/beef_intensities.csv --outLabels data/beef_labels.csv --dir "D:/workbench/data/spectro/[SpiderMass]-Databank_Synapt_Beef-liver_Reproducibility/20170228_Bl_Reproducibility/mzXML" --meta "D:\workbench\data\spectro\[SpiderMass]-Databank_Synapt_Beef-liver_Reproducibility\20170228_Bl_Reproducibility\20170209_Bl_Reproducibility-experiment_Sample-list.xlsx" --label_column Analysis.time

