# This Script changes the original format of county data 
# to our own format for deep learning....

### Erase variables and set path...
rm(list=ls()) 
 
# +++++ LIBRARIES +++++

# +++++ PARAMETERS +++++
nomFileOrig <- 'Datasets/extracloumn_WA.csv' 
nomFileFinal <- 'Datasets/Weekly_extracloumn_WA.csv' 
#NC_CasesCountyAugustTest26.csv
dayIni <- as.Date('04/19/2020', '%m/%d/%Y') 
dayFin <- as.Date('09/27/2020', '%m/%d/%Y') 

# +++++ MAIN PROCEDURE +++++ 
# --- Open file 
tblOrig <- read.csv(nomFileOrig, header = TRUE)  
# --- Vars' names
colsTitle <- colnames(tblOrig)[2:dim(tblOrig)[2]]
# --- Dates' names
rowsTitle <- format(seq(dayIni, dayFin, by=7), '%d/%m/%Y') 
# --- Create empty matrix
matFinal <- matrix(0, length(rowsTitle), length(colsTitle))
colnames(matFinal) <- colsTitle
rownames(matFinal) <- rowsTitle
# --- For each week
auxDate <- format(as.Date(tblOrig[,1]), '%d/%m/%Y')
for (k in 1:length(rowsTitle)) {
  # To find row in final matrix
  iFil <- match(rowsTitle[k],auxDate)
  # To compute weekly stats
  auxWeek <- tblOrig[(iFil-6):iFil,2:dim(tblOrig)[2]]
  for (ic in 1:length(colsTitle)) {
    matFinal[k,ic] <- mean(as.numeric(auxWeek[,ic])) 
  }# end For ic
  
}# End for

# --- Save file
write.csv(matFinal, file=nomFileFinal) 




