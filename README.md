# cell_counting 
## implemented by Yezhen Wang
### ENVIRONMENT
python 3.6<br>
numpy 1.15.4<br>
pytorch 1.0.1

### PROLOGUE
I coded two methods to count cell numbers, the first based on gradient vector field(you can see it in cell_counting_detection.py) and the second based on hierarchical clustering(you can see it in cell_counting_clustering.py), and I found the latter have better results.
the vedio which was tested comes from [here](https://www.youtube.com/watch?v=gEwzDydciWc)<br>
the test codes(only offer clustering based method, the other have rather worse results, so I don't give that part codes to you) was written in the jupyter notebook, you can see them in main.ipynb(the first four cells) 
### PREDICTION ON SHANGHAITECH_B DATASET
some samples here<br>
![avatar](/home/zzn/PycharmProjects/cell_tracking/sample_img/1.jpg)
![avatar](/home/zzn/PycharmProjects/cell_tracking/sample_img/1.jpg)
![avatar](/home/zzn/PycharmProjects/cell_tracking/sample_img/1.jpg)
![avatar](/home/zzn/PycharmProjects/cell_tracking/sample_img/1.jpg)
![avatar](/home/zzn/PycharmProjects/cell_tracking/sample_img/1.jpg)