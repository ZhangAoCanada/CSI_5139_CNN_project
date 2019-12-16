from glob import glob

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Input, MaxPooling2D, concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def pad_data(data, width=1280, height=384):
    r = data.shape[0]
    c = data.shape[1]

    deltar = 0
    deltac = 0
    if r < height:
        deltar = height - r
    if c < width:
        deltac = width - c
    data_pad = np.pad(data, ((0, deltar), (0, deltac)))
    return data_pad


def resize_data(data, width=1280, height=384):
    data = resize(data, (height, width, 1),
                  mode='constant', preserve_range=True)
    return data


def read_img(filepath, width=1280, height=384):
    img_list = []
    all_files = glob(filepath + "/*png")
    for i in range(len(all_files)):
        img = np.array(
            imread("./src/data_processing/test_in/" + str(i) + ".png"))
        img_pad = pad_data(img)
        img_pad = resize_data(img_pad)
        img_list.append(img_pad)
    return np.asfarray(img_list)


def read_gt(filepath, width=1280, height=384):
    gt_list = []
    all_files = glob(filepath+"/*npy")
    for i in range(len(all_files)):
        gt = np.load("./src/data_processing/test_out/" + str(i) + ".npy")
        gt_pad = pad_data(gt)
        gt_pad = resize_data(gt_pad)
        gt_list.append(gt_pad)
    return np.asfarray(gt_list)


img_list = read_img("./src/data_processing/test_in")
gt_list = read_gt("./src/data_processing/test_out")

print("Img List shape={}\tGt List shaple={}".format(
    img_list.shape, gt_list.shape))
print("Img's shape={}\t Gt's shaple={}".format(
    img_list[0].shape, gt_list[0].shape))


def conv2d_block(input_tensor, n_filters, kernel_size=3, BN=True):
        # First Layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal", padding="same")(input_tensor)
    if BN:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Second Layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal", padding="same")(input_tensor)
    if BN:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters=12, dropout=0.5, BN=True):
    c1 = conv2d_block(input_img, n_filters*1, kernel_size=3, BN=BN)
    p1 = MaxPooling2D((2, 2))(c1)
    # p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters*2, kernel_size=3, BN=BN)
    p2 = MaxPooling2D((2, 2))(c2)
    # p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters*4, kernel_size=3, BN=BN)
    p3 = MaxPooling2D((2, 2))(c3)
    # p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters*8, kernel_size=3, BN=BN)
    p4 = MaxPooling2D((2, 2))(c4)
    # p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters*16, kernel_size=3, BN=BN)

    u6 = Conv2DTranspose(n_filters*8, (3, 3),
                         strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    # u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters *
                      8,  kernel_size=3, BN=BN)

    u7 = Conv2DTranspose(n_filters*4, (3, 3),
                         strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    # u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters *
                      4,  kernel_size=3, BN=BN)

    u8 = Conv2DTranspose(n_filters*2, (3, 3),
                         strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    # u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters *
                      2,  kernel_size=3, BN=BN)

    u9 = Conv2DTranspose(n_filters*1, (3, 3),
                         strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    # u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters,  kernel_size=3, BN=BN)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    model.compile(optimizer=Adam(), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def create_model():
    im_width = 1280
    im_height = 384
    input_img = Input((im_height, im_width, 1), name='img')
    model = get_unet(input_img)
    return model


model = create_model()
model.summary()

# loss, acc = model.evaluate(img_list,  gt_list, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


# Loads the weights
model.load_weights("modelfolder/model-tgs-salt-2.h5")

# Predict on train, val and test
preds_test = model.predict(img_list, verbose=1)
preds_test = (preds_test > 0.5).astype(np.uint8)

# ix = 12
# intersection = np.logical_and(gt_list[ix,...,0], preds_test[ix,...,0])
# union = np.logical_or(gt_list[ix,...,0], preds_test[ix,...,0])
# iou_score = np.sum(intersection) / np.sum(union)
intersection = np.logical_and(gt_list, preds_test)
union = np.logical_or(gt_list ,preds_test)
iou_score = np.sum(intersection) / np.sum(union)
print(iou_score)



# Threshold predictions
# preds_test = (preds_test > 0.5).astype(np.uint8)
# print(preds_test.shape)
# # Select random index from trainning set.
# for i in range(5): 
#     ix = random.randint(0, 19)
#     has_mask = preds_test[ix].max() > 0
#     fig, ax = plt.subplots(2, 2)
#     print("show img {} and its mask".format(ix))
#     ax[0,0].imshow(img_list[ix, ..., 0], cmap='seismic')
#     if has_mask:
#         ax[0,0].contour(preds_test[ix].squeeze(), colors='k', levels=[0.5])
#     ax[0,0].set_title('Seismic')

#     ax[0,1].imshow(preds_test[ix].squeeze(),  cmap='gray')
#     ax[0,1].set_title('test')


#     ax[1,0].imshow(img_list[ix, ..., 0], cmap='seismic')
#     if has_mask:
#         ax[1,0].contour(gt_list[ix].squeeze(), colors='k', levels=[0.5])
#     ax[1,0].set_title('Seismic')

#     ax[1,1].imshow(gt_list[ix].squeeze(),  cmap='gray')
#     ax[1,1].set_title('ground truth')
#     show()
# Re-evaluate the model
# loss, acc = model.evaluate(img_list,  gt_list, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
