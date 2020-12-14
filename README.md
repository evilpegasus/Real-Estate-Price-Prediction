# Real Estate Price Prediction
### Berkeley SAAS CX Fall 2020 Kaggle Competition
### Ming Fong and Yifan Zhang
Winning solution

Predicting real estate sale prices using property data.

## Data
Data can be downloaded from the [Kaggle competition data page](https://www.kaggle.com/c/saas-2020-fall-cx-kaggle-compeition/data).
In the repo, data is in the `/data` directory.

There are 3 data files:
* [data/test_features.csv](data/test_features.csv)
* [data/train_features.csv](data/train_features.csv)
* [data/train_targets.csv](data/train_features.csv)

[output/sample_submission.csv](output/sample_submission.csv) is an example of a file that is ready to submit to Kaggle. There are two columes: `id` and `SALE PRICE`.

## Kaggle Link
https://www.kaggle.com/c/saas-2020-fall-cx-kaggle-compeition


## Notes

Building codes 

https://www1.nyc.gov/assets/finance/jump/hlpbldgcode.html


remove outliers
check negative price predicitons


Check if building or tax class changes
    could mean redeveloped housing
    add column "classChanged" - 1 if yes, 0 if no

Check if apartment number is present
    add column "hasApartmentNumber" - 1 or 0
