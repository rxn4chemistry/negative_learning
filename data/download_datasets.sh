# Downloading datasets
mkdir -p data/source_data

# USPTO
mkdir data/source_data/uspto
wget https://figshare.com/ndownloader/articles/5104873/versions/1 -O data/source_data/uspto.zip
unzip data/source_data/uspto.zip -d data/source_data/uspto
rm data/source_data/uspto.zip
7za e data/source_data/uspto/1976_Sep2016_USPTOgrants_smiles.7z -o"data/source_data/uspto"
rm data/source_data/uspto/*7z data/source_data/uspto/*zip

# REGIOSQM
mkdir data/source_data/regiosqm
wget  https://raw.githubusercontent.com/jensengroup/SI_RegioSQM20/main/RegioSQM20/compounds.smiles -O data/source_data/regiosqm/compounds_smiles.csv
extract-regiosqm-dataset
