# **Artistic-style-text-detection**
* Artistic-style text detector and a new Movie-Poster dataset(https://arxiv.org/pdf/2406.16307)
## **Guidelines of the Movie-Poster Dataset**
### Summarize
We collected 1,500 movie posters featuring various artistic-style titles to address the current market’s lack of artistic-style text data. It contains 1500 images, of which 1100
are in the training set and 400 in the testing set.
### Statistic

| Complexity distribution | Launage type distribution | Text type distribution |  
| - | - | - | 
|Easy  34.6%|English  32.4%|Highly similar to the background  23.5%|
|Lightly complex  28.0%|Chinese  43.0%|Extreme aspect ratios  36.7%|
|Moderately complex  21.3%|English and Chinese  24.1%|Extreme abstraction  23.4%|
|Extremely complex  16.1%|Others  0.5%|Others  16.4%|

### Annotation process and text types
<img src="https://github.com/AXNing/Artistic-style-text-detection/blob/main/sample.png" width="80%"/>

### Framework

```
Movie-Poster/
│
├── Train/
│   ├── img/
│   └── gt/
│
└── Test/
    ├── img/
    └── gt/
```
### Download
The Movie-Poster is available at [Movie-Poster](https://drive.google.com/file/d/1anlWPsCX-6aYhUDqC33SXRufcpPpjLE2/view?usp=drive_link).

### Partial sample visualization
<img src="https://github.com/biedaxiaohua/Artistic-style-text-detection/blob/main/visualization1.png" width="80%"/>

### Challenge
We hope to attract more researchers to address the current difficulties in artistic-style text detection.
Visualization of selected failure cases:
<img src="https://github.com/AXNing/Artistic-style-text-detection/blob/main/Failcases.png" width="100%"/>

## **Artistic_style text detection**
### Prerequisites
  python >= 3.7 ;  
  PyTorch >= 1.7.0;   
  Numpy >=1.2.0;   
  CUDA >=11.1;  
  GCC >=10.0;   
  Opencv-python < 4.5.0
### Testing
#### Prepare dataset
```
data/
│
├── Movie-Poster/
│
└── total-text/
│
│
```
#### Download weight
Click on this [link](https://drive.google.com/file/d/1VpIeC7Dkv-5Ywnt4_7e8uvpCNjmtwMR2/view?usp=sharing) to download it and place it in the specified folder.

#### Running the Testing scripts

```
cd scripts-eval/
sh Eval_Movie_Poster.sh
```
#### Visual comparison
<img src="https://github.com/AXNing/Artistic-style-text-detection/blob/main/duibi.png" width="80%"/>

#### Visualization of results for other datasets
<img src="https://github.com/AXNing/Artistic-style-text-detection/blob/main/or.png" width="60%"/>
<img src="https://github.com/AXNing/Artistic-style-text-detection/blob/main/curv.png" width="60%"/>
