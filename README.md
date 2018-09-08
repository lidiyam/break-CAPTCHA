# break-captcha

Solving simple 4-letter captcha with OpenCV and Keras.

### Intructions

Clone the repo & build a container:
```
docker build .

docker run -it <container_id> bash
```

From inside the container:
```
python3 train_model.py

python3 break_captcha.py --img <PATH_TO_THE_IMAGE_FILE>
```

###

Starter code from [how to break captcha system with ML](https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710)
