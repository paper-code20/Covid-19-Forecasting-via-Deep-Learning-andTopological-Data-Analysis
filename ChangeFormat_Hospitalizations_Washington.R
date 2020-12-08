# This Script changes the original format of county data 
# to our own format for deep learning....

### Erase variables and set path...
rm(list=ls()) 
 
# +++++ LIBRARIES +++++

# +++++ PARAMETERS +++++
nomFileOrig <- 'Datasets/Washington_PUBLIC_CDC_Event_Date_SARS.csv' 
nomFileFinal <- 'Datasets/Washington_Hospitalization.csv'
#NC_CasesCountyAugustTest26.csv
dayIni <- as.Date('04/15/2020', '%m/%d/%Y') 
dayFin <- as.Date('09/30/2020', '%m/%d/%Y') 

# +++++ MAIN PROCEDURE +++++ 
# --- Open file 
tblOrig <- read.csv(nomFileOrig, header = TRUE)  
# --- Counties' names
nomCounty <- sort(as.character(unique(tblOrig[,1])))
colsTitle <- c()
for(i in 1:length(nomCounty)) {
  colsTitle[i] <- substr(nomCounty[i],1,nchar(nomCounty[i])-7)
} # End for
# --- Dates' names
rowsTitle <- sort(unique(as.Date(tblOrig[,2])))
rowsTitle <- rowsTitle[rowsTitle>=dayIni]
rowsTitle <- rowsTitle[rowsTitle<=dayFin]
rowsTitle <- format(rowsTitle, '%d/%m/%Y')
# --- Create empty matrix
matFinal <- matrix(0, length(rowsTitle), length(colsTitle))
colnames(matFinal) <- colsTitle
rownames(matFinal) <- rowsTitle
# --- Auxiliar variables
auxCounty <- as.character(tblOrig[,1])
#auxDate <- format(as.Date(tblOrig[,2], '%m/%d/%Y'), '%m/%d/%Y')
auxDate <- format(as.Date(tblOrig[,2]), '%d/%m/%Y')
auxHospi <- as.numeric(tblOrig[,3])
# --- Fill out using original data
for (k in 1:dim(tblOrig)[1]) {
  # To find coordinates in final matrix
  iFil <- match(auxDate[k],rowsTitle)
  iCol <- match(auxCounty[k],nomCounty)
  # To save data
  matFinal[iFil, iCol] <- auxHospi[k] 
} # End for Fill out

# Delete Unassigned
matFinal <- matFinal[,1:(dim(matFinal)[2]-1)]

# --- Save file
write.csv(matFinal, file=nomFileFinal) 




