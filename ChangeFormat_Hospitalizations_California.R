# This Script changes the original format of county data 
# to our own format for deep learning....

### Erase variables and set path...
rm(list=ls()) 

# +++++ LIBRARIES +++++

# +++++ PARAMETERS +++++
nomFileOrig <- 'Datasets/California_hospitals_by_county.csv' 
nomFileFinal <- 'Datasets/California_Hospitalization.csv' 
#NC_CasesCountyAugustTest26.csv 
dayIni <- as.Date('04/15/2020', '%m/%d/%Y') 
dayFin <- as.Date('09/30/2020', '%m/%d/%Y') 

# +++++ MAIN PROCEDURE +++++ 
# --- Open file 
tblOrig <- read.csv(nomFileOrig, header = TRUE)  
# --- Counties' names 
nomCounty <- as.character(unique(tblOrig[,1]))
colsTitle <- sort(nomCounty) 
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
auxDate <- format(as.Date(tblOrig[,2]), '%d/%m/%Y')
auxHospi <- as.numeric(tblOrig[,3])+as.numeric(tblOrig[,4]) 
# --- Fill out using original data 
for (k in 1:dim(tblOrig)[1]) {
  # To find coordinates in final matrix
  iFil <- match(auxDate[k],rowsTitle)
  iCol <- match(auxCounty[k],colsTitle)
  # To save data
  matFinal[iFil, iCol] <- auxHospi[k] 
} # End for Fill out

# --- Save file
write.csv(matFinal, file=nomFileFinal) 




