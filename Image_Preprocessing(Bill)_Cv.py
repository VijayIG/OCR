#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib
matplotlib.rcParams['figure.figsize'] = (30.0, 20.0)
matplotlib.rcParams['image.cmap'] = 'gray'


# In[3]:


image = cv2.imread("MicrosoftTeams-image (7).png")
plt.imshow(image)


# In[4]:


imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(imageGray)
print(imageGray.shape)


# In[5]:


#Binary
img_bin = cv2.threshold(imageGray, 127, 255, cv2.THRESH_BINARY)[1]
plt.imshow(img_bin)


# # Contours

# In[6]:


contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
len(contours)


# In[7]:


imagecopy=image.copy()
final=cv2.drawContours(imagecopy, contours, -1, (0,255,0), 3);
plt.imshow(final,cmap="gray")


# In[8]:


selected_contours = []
new=[]


for cnt in contours:
    area = cv2.contourArea(cnt)
    new.append(area)
    if area==1879315.5:
        selected_contours.append(cnt)
#     print(area)
    
# print(len(area))
# print(selected_contours)


# In[24]:


cont1 = selected_contours[0]

x,y,w,h = cv2.boundingRect(cont1)

print(x,y,w,h)

img_sub = image[y:y+h,x:x+w]

plt.imshow(img_sub)


# # Gray_scale

# In[10]:


imageGray = cv2.cvtColor(img_sub,cv2.COLOR_BGR2GRAY)
plt.imshow(imageGray)
print(imageGray.shape)


# # Thershold

# In[11]:


#Binary
img_bin = cv2.threshold(imageGray, 127, 255, cv2.THRESH_BINARY)[1]
plt.imshow(img_bin)


# In[12]:


plt.figure()
plt.subplot(121)
plt.imshow(img_sub)
plt.title("Original Image");
plt.subplot(122)
plt.imshow(img_bin)
plt.title("Grayscale Image");


# In[13]:


#OTSU
img_bin = cv2.threshold(imageGray, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
# plt.imshow(img_bin)


# # Tesseract

# In[14]:


import pytesseract
from typing import OrderedDict


# In[15]:


image_data = pytesseract.image_to_string(img_bin,config="--psm 6")
print(image_data)


# In[16]:


import re


# In[17]:


pattern=re.compile(r'\d+\.\d+\s')
regex_value=re.findall(pattern,image_data)
print(regex_value)


# In[18]:


pattern=re.compile(r'\*\s\d+\s?[A-za-z]+\s?[A-za-z]+\s?[A-za-z]+|\*\s[a-zA-z]+\s[A-Za-z]+\s?[A-za-z]+|\%\s\d+\s?[A-za-z]+\s?[A-za-z]+\s?[A-za-z]+|\%\s[A-za-z]+\s[A-za-z]+\s[0-9]+/[0-9]+\s[A-za-z]+|[a-z]+\s[A-z]+\s[A-Z]+\s[A-Za-z]+')
regex=re.findall(pattern,image_data)
print(regex)
# print(regex[0][0])
# regex=str(regex)
# print(regex)
# print(type(regex))


# In[19]:


new=[]
for i in regex:
    if i.count(i[0])==1:
        new.append(i.replace(i[0],""))
    else:
        new.append(i)
print(new) 


# In[20]:


output=zip(new,regex_value)
result=dict(output)
print(result)


# In[21]:


print(result)


# In[22]:


import json
with open("extraction_bill.json", "w") as write_file:
    json.dump(result, write_file)


# In[ ]:




