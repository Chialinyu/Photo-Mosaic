#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cherrypy, os, urllib, pickle
from numpy import *

import numpy as np
from scipy.fftpack import fft, ifft, dct
from PIL import Image
import os.path
import operator
from collections import Counter
from numpy import array
from scipy.cluster.vq import vq,kmeans, kmeans2, whiten

# ---------------------- 設定 ---------------------- #
query_file_number = 0 # 要被搜尋的圖片編號（請輸入 0 ~ 999）
fileCount = 1000 # 要被比較的圖片數量（請輸入1~1000）
# -------------------------------------------------- #

QueryFile = "ukbench0"+str(query_file_number).zfill(4)
QueryFilename = QueryFile+".jpg"
img = Image.open("./dataset/"+QueryFilename)
width, height = img.size

def color_Average(img):
	width, height = img.size
	r, g, b = 0, 0, 0
	count = 0
	pixel = img.load()
	for i in xrange(width):
		for j in xrange(height):
			R, G, B = pixel[i,j]
			r += R
			g += G
			b += B
			count +=1
	return ((r/count), (g/count), (b/count))

def RBG_Average(img):
	# Q0 RGB Average
	dis = np.zeros((fileCount,2),int)
	dis_0 = np.zeros((fileCount,2),int)
	tmpSave0 = np.zeros((fileCount,3),int)
	if os.path.isfile("Feature_RGB"):
		print "RGB Average Feature file is exit."
	else:
		for file_name in xrange(fileCount):
			img_query = Image.open("./dataset/ukbench0"+str(file_name).zfill(4)+".jpg")
			img_query = img_query.resize((width,height))
			tmpSave0[file_name] = color_Average(img_query)
			tmpSave0.tofile("Feature_RGB")

	tmpSave0 = np.fromfile("Feature_RGB", dtype=np.int)
	tmpSave0 = tmpSave0.reshape((fileCount,3))
	img = img.resize((width,height))
	rgb_q = color_Average(img)

	for file_name in xrange(fileCount):
		rgb_query = tmpSave0[file_name]
		for i in xrange(len(rgb_query)):
			dis[file_name][0] = file_name
			dis[file_name][1] += (rgb_q[i]-rgb_query[i])**2

	dis_0 = sorted(dis, key = lambda x : x[1])
	return dis_0[0][0]
	
def Color_Historgam(img):
	# Q1 Color Histogram
	img_q1 = img.copy()

	dis = np.zeros((fileCount,2),int)
	dis_1 = np.zeros((fileCount,2),int)
	tmpSave = np.zeros((fileCount,768),int)
	if os.path.isfile("Feature_ColorHistogram"):
		print "Color Histogram Feature file is exit."
	else:
		for file_name in xrange(fileCount):
			img_query = Image.open("./dataset/ukbench0"+str(file_name).zfill(4)+".jpg")
			img_query = img_query.resize((width,height))
			tmpSave[file_name] = img_query.histogram()
		tmpSave.tofile("Feature_ColorHistogram")

	tmpSave = np.fromfile("Feature_ColorHistogram", dtype=np.int)
	tmpSave = tmpSave.reshape((fileCount,768))

	hist_q1 = img.resize((width,height)).histogram()

	for file_name in xrange(fileCount):
		hist_query = tmpSave[file_name]
		for i in xrange(len(hist_query)):
			dis[file_name][0] = file_name
			dis[file_name][1] += (hist_q1[i]-hist_query[i])**2

	dis_1 = sorted(dis, key = lambda x : x[1])
	return dis_1[0][0]

# Q2 Color Layout

def zigzag(arr):
	new = []
	i = 0
	j = 0
	up=1
	turned = False
	d =  [[1, -1],[-1, 1]]
	corner = [[ 1, 0, 0, 1 ], [ 0, 1, 1, 0 ]]
	m = 8
	n = 8
	while i < m and j < n:
		new.append(arr[j][i])
		if i == 0 or j == 0 or i == m - 1 or j == n - 1:
			if not turned :
				k = 2 * (up * (j / (n - 1)) | (1 - up) * (i / (m - 1)));
				i += corner[up][k];
				j += corner[up][k + 1];
				turned = True;
				up = 1 - up;
				continue;
			else:
				turned = False
		i += d[up][0]
		j += d[up][1]
	return new

def color_layout(img2):
	blocksize = 8
	block_w = width/blocksize
	block_h = height/blocksize

	color = np.zeros((blocksize,blocksize,3),int)
	for i in xrange(0,blocksize):
		for j in xrange(0,blocksize):
			box = (i*block_w, j*block_h, (i+1)*block_w, (j+1)*block_h)
			tmp = img2.crop(box)
			color[i][j][0], color[i][j][1], color[i][j][2] = color_Average(tmp)

	img_q2_64 = Image.new("RGB", (blocksize, blocksize))
	pixel_q2_64 = img_q2_64.load()
	for i in xrange(blocksize):
		for j in xrange(blocksize):
			pixel_q2_64[i,j] = (color[i][j][0], color[i][j][1], color[i][j][2])

	img_q2_64_ycbcr = img_q2_64.convert('YCbCr')

	img_q2_64_ycbcr_fft = dct(dct(img_q2_64_ycbcr, norm='ortho'), norm='ortho')
	img_q2_64_ycbcr_fft_sort = zigzag(img_q2_64_ycbcr_fft)

	return img_q2_64_ycbcr_fft_sort

def color_layout_dis(img):
	dis1 = np.zeros((fileCount,2),float)
	dis1_1 = np.zeros((fileCount,2),float)

	tmpSave2 = np.zeros((fileCount,64,3),float)
	if os.path.isfile("Feature_ColorLayout"):
		print "Color Layout Feature file is exit."
	else:
		for file_name in xrange(fileCount):
			img_query = Image.open("./dataset/ukbench0"+str(file_name).zfill(4)+".jpg")
			img_query = img_query.resize((width,height))
			tmpSave2[file_name] = color_layout(img_query)

		tmpSave2.tofile("Feature_ColorLayout")

	tmpSave2 = np.fromfile("Feature_ColorLayout")
	tmpSave2 = tmpSave2.reshape((fileCount,64,3))

	img = img.resize((width,height))
	a = color_layout(img)

	for file_name in xrange(fileCount):
		b = tmpSave2[file_name]
		for j in xrange(3):
			tmp = 0
			for i in xrange(len(a)):
				dis1[file_name][0] = file_name
				tmp += (a[i][j]-b[i][j])**2
			dis1[file_name][1] += tmp**(0.5)

	dis1_1 = sorted(dis1, key = lambda x : x[1])
	return int(dis1_1[0][0])



def mosaic(img0,thumbsize,featureFunction):
	thumb_w = width/thumbsize
	thumb_h = height/thumbsize

	for i in xrange(0,thumbsize):
		for j in xrange(0,thumbsize):
			box = (i*thumb_w, j*thumb_h, (i+1)*thumb_w, (j+1)*thumb_h)
			tmp = img0.crop(box)
			if featureFunction == 0:
				similarNumber = RBG_Average(tmp)
			if featureFunction == 1:
				similarNumber = Color_Historgam(tmp)
			if featureFunction == 2:
				similarNumber = color_layout_dis(tmp)
			tmp = Image.open("./dataset/ukbench0"+str(similarNumber).zfill(4)+".jpg").resize((thumb_w,thumb_h))
			img0.paste(tmp,(i*thumb_w, j*thumb_h, (i+1)*thumb_w, (j+1)*thumb_h))
			print ((i)*thumbsize+j+1)*100/(thumbsize*thumbsize),"%"
			
	img0.save("tmp.jpg")

# makes the index URL accessible and the last line starts the CherryPy web server with configurations read from service.conf.

class SearchDemo:
	def __init__(self):
		# load list of images
		now_path =  os.path.dirname(os.path.realpath(__file__))
		# self.imlist = [now_path+'/dataset/ukbench00000.jpg',now_path+'/dataset/ukbench00001.jpg',now_path+'/dataset/ukbench00002.jpg',now_path+'/dataset/ukbench00003.jpg',now_path+'/dataset/ukbench00004.jpg',now_path+'/dataset/ukbench00005.jpg',now_path+'/dataset/ukbench00006.jpg',now_path+'/dataset/ukbench00007.jpg',now_path+'/dataset/ukbench00008.jpg',now_path+'/dataset/ukbench00009.jpg',now_path+'/dataset/ukbench00010.jpg']
		self.imlist = [now_path+'/dataset/ukbench00000.jpg']
		for i in xrange(1,1000):
			self.imlist.append(now_path+"/dataset/ukbench0"+str(i).zfill(4)+".jpg")
		# print self.imlist, len(self.imlist)
		self.nbr_images = len(self.imlist)
		self.ndx = range(self.nbr_images)


		# set max number of results to show
		self.maxres = 5

		# header and footer html
		self.header = """
		<!doctype html>
			<head>
			<title>Carolyn's Photo Mosaic &hearts;</title>
			<script>
				function go(){
					featureID = document.getElementById('selfeature').value;

					if (document.getElementById('blockradio8').checked) {
						blockscount = document.getElementById('blockradio8').value;
					}else if (document.getElementById('blockradio20').checked) {
						blockscount = document.getElementById('blockradio20').value;
					}else if (document.getElementById('blockradio40').checked) {
						blockscount = document.getElementById('blockradio40').value;
					}else if (document.getElementById('blockradio80').checked) {
						blockscount = document.getElementById('blockradio80').value;
					}else if (document.getElementById('blockradio160').checked) {
						blockscount = document.getElementById('blockradio160').value;
					}

					org = window.location.href;
					org = org.split("&");
					org = org[0];
					window.location.href = org +
										"&featureID=" + featureID +
										"&blockscount=" + blockscount;
				}
				
			</script>
			<style>
				html, body {
					/*height: 100%; margin:0;*/
				}
				body {
					width:80%;
					margin-left: auto;
					margin-right: auto; 
					font-family: 'Open Sans', sans-serif;
					font-weight: 400;
					font-size: 1em;
					color: #8a8683;
					/*background-color:#ffffff;*/
					background: url(http://subtlepatterns2015.subtlepatterns.netdna-cdn.com/patterns/squared_metal.png)repeat;
				}
				header {
					width: 80%;
					left:0; right:0;
					/*background:url(http://subtlepatterns2015.subtlepatterns.netdna-cdn.com/patterns/tweed.png)repeat;*/
					border-radius: 10px;
				}
				.main{
					background-color:#ffffff; 
					margin:auto;
					padding: 20px;
					border-radius: 5px;
				}
				.left{
					width:70%;
					/*float: left;*/
					display: inline-block;
				}
				.right{
					width:25%;
					float: right;
					display: inline-block;
				}
				img {
					max-width: 100%;
					margin: 5px;
					box-sizing:border-box;
					border-radius: 5px;
				}
	
				.imgBox { width: 120px;}
				.imgBox:hover {box-shadow: 0 0 10px #8a8683; }
				a:link {
					color: #f68f67;
					text-decoration: none;
				}
				a:hover {
					color: #bde2df;
					text-decoration: none;
				}
				a:visited {
					color: #f68f67;
					text-decoration: none;
				}
				footer{
					clear:both;
					text-align: center;
					color:white;
					/*background-color:#8a8683;*/
					padding: 10px 0px 10px 0px;
				}
			</style>
			</head>
			<body>
			"""
		self.footer = """
			</body>
			<footer>
				<hr>
				<div><a href="http://www.cs.nccu.edu.tw/">NCCUCS</a> MMHW4 | <a href="https://www.facebook.com/carolyn.yu.54" target="_blank">Carolyn</a> &copy; 2016 All Rights Reserved
				</div>
			</footer>
			</html>
			"""

	def index(self,query=None,featureID=None,blockscount=None):
		html = self.header
		html += """
			<header>
			<div class='top'>
			<h1>104多媒體 #HW4 Photo Mosaic &hearts;</h1>
			<h2>104753007 資科碩ㄧ 游佳霖</h2>
			</div>

			</header>
			<div class='main'>
			<div class='left'>
			
			<h3>Step1: </h3>
			Click an image to search.
			"""
		html += """
			<a href='?query='> Random selection </a> of images.
			<br /><br />
			"""
		if query:
			q = query.decode('utf-8').encode('gb18030')
			for i in self.ndx[:self.maxres]:
				imname = self.imlist[i]
				html += "<a href='?query="+imname+"'>"
				html += "<img class='imgBox' src='"+imname+"' width='100' />"
				html += "</a>"
			
			if featureID:
				if  blockscount:
					img0 = Image.open(q)
					thumbsize = int(blockscount)
					featureFunction = int(featureID)
					mosaic(img0,thumbsize,featureFunction)
					html += "<h3>Result: </h3><img src='"+os.path.dirname(os.path.realpath(__file__))+'/tmp.jpg'+"' width='640px' />"
			else:	
				html += "<h3>Result: </h3><img src='"+q+"' width='640px' />"
			html += "</div>" 
			html += "<div class='right'><br><h3>Chose image:</h3>"
			html += "<img src='"+q+"' width='200px' />"
		else:
			random.shuffle(self.ndx)
			for i in self.ndx[:self.maxres]:
				imname = self.imlist[i]
				html += "<a href='?query="+imname+"'>"
				html += "<img class='imgBox' src='"+imname+"' width='100' />"
				html += "</a>"
			html += "<h3>Result: </h3><img src='"+os.path.dirname(os.path.realpath(__file__))+'/dataset/ukbench00000.jpg'+"' width='640px' />"
			html += "</div>" 
			html += "<div class='right'><br><h3>Chose image:</h3>"
			html += "<img src='"+os.path.dirname(os.path.realpath(__file__))+'/dataset/ukbench00000.jpg'+"' width='200px' />"

	
		html +=	"""
			<div>
			<br><br>
			<h3>Step2:</h3>
			<h3>Choose the Feature:</h3>
			<select id="selfeature">
				<option value='0'>RGB Average</option>
				<option value='1'>Color Histogram</option>
				<option value='2'>Color Layout</option>
			</select>
			</div>
			<br>
			<h3>Choose the block counts:</h3>
			<div>
			<input id="blockradio8" type="radio" name="entry.1.group" value="8" checked=true>8x8<br>
			<input id="blockradio20" type="radio" name="entry.1.group" value="20">20x20<br>
			<input id="blockradio40" type="radio" name="entry.1.group" value="40">40x40<br>
			<input id="blockradio80" type="radio" name="entry.1.group" value="80">80x80<br>
			<input id="blockradio160" type="radio" name="entry.1.group" value="160">160x160<br>
			</div>
			<h3>Step3: Submit!</h3>
			<input type="submit" onclick="go()" name="submit" value="OK&nbsp;&rsaquo;&rsaquo;">
			</div>
			""" 
		html += "</div>"   
		html += self.footer
		return html
	
	index.exposed = True

cherrypy.quickstart(SearchDemo(), '/', config=os.path.join(os.path.dirname(__file__), 'service.conf'))
