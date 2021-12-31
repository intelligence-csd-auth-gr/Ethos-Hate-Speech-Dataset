# ETHOS Hate Speech Dataset
ETHOS: multi-labEl haTe speecH detectiOn dataSet. This repository contains a dataset for hate speech detection on social media platforms, called Ethos. There are two variations of the dataset:
- Ethos_Dataset_Binary.csv[Ethos_Dataset_Binary.csv] contains 998 comments in the dataset alongside with a label about hate speech *presence* or *absence*. 565 of them do not contain hate speech, while the rest of them, 433, contain. 
- Ethos_Dataset_Multi_Label.csv [Ethos_Dataset_Multi_Label.csv] which contains 8 labels for the 433 comments with hate speech content. These labels are *violence* (if it incites (1) or not (0) violence), *directed_vs_general* (if it is directed to a person (1) or a group (0)), and 6 labels about the category of hate speech like, *gender*, *race*, *national_origin*, *disability*, *religion* and *sexual_orientation*.

## Ethos /ˈiːθɒs/ 
is a Greek word meaning “character” that is used to describe the guiding beliefs or ideals that characterize a community, nation, or ideology. The Greeks also used this word to refer to the power of music to influence emotions, behaviors, and even morals.

Please check our older dataset as well: https://intelligence.csd.auth.gr/topics/hate-speech-detection/

## Reference
Please if you use this dataset in your research cite out preprint paper: [ETHOS: an Online Hate Speech Detection Dataset](https://arxiv.org/abs/2006.08328)
```
@misc{mollas2020ethos,
    title={ETHOS: an Online Hate Speech Detection Dataset},
    author={Ioannis Mollas and Zoe Chrysopoulou and Stamatis Karlos and Grigorios Tsoumakas},
    year={2020},
    eprint={2006.08328},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Contributors on Ethos
Name | Email
--- | ---
Grigorios Tsoumakas | greg@csd.auth.gr
Ioannis Mollas | iamollas@csd.auth.gr
Zoe Chrysopoulou | zoichrys@csd.auth.gr
Stamatis Karlos | stkarlos@csd.auth.gr

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
