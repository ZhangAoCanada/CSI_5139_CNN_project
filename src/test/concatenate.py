import cv2
img1 = cv2.imread("/Users/xtstc131/Documents/CSI_5139_CNN_project/unet_19.png")
img2 = cv2.imread("/Users/xtstc131/Documents/CSI_5139_CNN_project/unet_50.png")
img3 = cv2.imread("/Users/xtstc131/Documents/CSI_5139_CNN_project/unet_100.png")
img4 = cv2.imread("/Users/xtstc131/Documents/CSI_5139_CNN_project/unet_102.png")
img5 = cv2.imread("/Users/xtstc131/Documents/CSI_5139_CNN_project/unet_150.png")

img1 = cv2.resize(img1,(1280, 384))
img2 = cv2.resize(img2,(1280, 384))
img3 = cv2.resize(img3,(1280, 384))
img4 = cv2.resize(img4,(1280, 384))
img5 = cv2.resize(img5,(1280, 384))


_img1 = cv2.imread("/Users/xtstc131/Downloads/convsegnet_dice_19.png")
_img2 = cv2.imread("/Users/xtstc131/Downloads/convsegnet_dice_50.png")
_img3 = cv2.imread("/Users/xtstc131/Downloads/convsegnet_dice_100.png")
_img4 = cv2.imread("/Users/xtstc131/Downloads/convsegnet_dice_102.png")
_img5 = cv2.imread("/Users/xtstc131/Downloads/convsegnet_dice_150.png")

cat1 = im_h = cv2.hconcat([_img1,img1 ])
cat2 = im_h = cv2.hconcat([_img2,img2 ])
cat3 = im_h = cv2.hconcat([_img3,img3 ])
cat4 = im_h = cv2.hconcat([_img4,img4 ])
cat5 = im_h = cv2.hconcat([_img5,img5 ])

vcat = cv2.vconcat([cat1,cat2,cat3,cat4,cat5])
cv2.imwrite("0.png",vcat)
# cv2.imwrite("2.png",cat2)
# cv2.imwrite("3.png",cat3)
# cv2.imwrite("4.png",cat4)
# cv2.imwrite("5.png",cat5)

