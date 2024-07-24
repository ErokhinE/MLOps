# Directory for configs of Hydra

configs
├── config.yaml   #main config for sampling and versioning the data
├── data_version.yaml  #config to store the data version
├── evaluate.yaml      #config to evaluate models
├── main.yaml          #config for parallel training 
└── model
    ├── gradient_boosting.yaml   #config to train the gradient_boosting regressor
    ├── model.yaml               #base config for training
    └── random_forest.yaml       #config to train the random_forest regressor
