import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import cv2
import os
import re
import sys



TARGET_DIR = "data/target"
RENDER_DIR = "data/render"
RENDER_STR = "{}.png"
DEBUG = False

class Render:
	def __init__(self, font=None, symbol=None):
		self.T0 = np.eye(3, 3)
		self.S0 = np.eye(3, 3)
		self.Tc = np.eye(3, 3)
		self.St = np.eye(3, 3)
		self.Tt = np.eye(3, 3)
		self.Rt = np.eye(3, 3)
		self.sigma = 0.01
		self.img = None
		self.target_shape = None
		self.box = None
		self.font = font
		self.symbol = symbol
		self.angle = 0.0
		if (font != None) and (symbol != None):
			self.load_symbol(font, symbol)

	def load_file(self, img_file):
		#print("Load image %s" % img_file)
		self.img = cv2.imread(img_file, 0)
		self.img = cv2.GaussianBlur(self.img, (11,11), 0)
		assert type(self.img) == np.ndarray

	def load_symbol(self, font, symbol):
		img_name = RENDER_STR.format(symbol)
		img_file = os.path.join(RENDER_DIR, font, img_name)
		self.load_file(img_file)

	def set_shape(self, target_shape):
		f,c = target_shape
		assert f > 0 and c > 0
		self.target_shape = target_shape

	def show(self, img = None):
		if type(img) != np.ndarray:
			img = self.img

		assert type(img) == np.ndarray
		view = pg.GraphicsView()

		l = pg.GraphicsLayout(border=(100,100,100))
		view.setCentralItem(l)
		view.show()
		ii = pg.ImageItem(img.astype(np.float).T)
		vb = l.addViewBox(lockAspect=True, invertY=True)
		vb.addItem(ii)
		QtGui.QApplication.exec_()

	def box_sobel(self, img):
		SOBEL_K = 3
		K = 0.1

		sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=SOBEL_K).astype(np.float)
		sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=SOBEL_K).astype(np.float)

		fil, col = img.shape
		maskx = 1 + np.tile(np.arange(col), (fil,1))
		maskx_ = maskx[::,::-1]
		masky = 1 + np.tile(np.arange(fil).reshape((fil,1)), (1, col))
		masky_ = masky[::-1,::]

		distx  = np.absolute(sobelx) * np.exp(-maskx*K)
		distx_ = np.absolute(sobelx) * np.exp(-maskx_*K)
		disty  = np.absolute(sobely) * np.exp(-masky*K)
		disty_ = np.absolute(sobely) * np.exp(-masky_*K)

		_, _, _, x0 = cv2.minMaxLoc(distx)
		_, _, _, x1 = cv2.minMaxLoc(distx_)
		_, _, _, y0 = cv2.minMaxLoc(disty)
		_, _, _, y1 = cv2.minMaxLoc(disty_)

		x0y, x0x = x0
		x1y, x1x = x1
		y0y, y0x = y0
		y1y, y1x = y1
		box = (x0y, x1y+1, y0x, y1x+1)
		return box

	def bound_render(self):
		x0, x1, y0, y1 = self.box_sobel(self.img)
		self.box = (x0, x1, y0, y1)
		self.render_shape = (y1-y0+2, x1-x0+2)
		#print("Render shape = {}".format(self.render_shape))
		ht,wt = self.target_shape
		hr,wr = self.render_shape

		#pt = float(wt) / float(ht)
		pr = float(wr) / float(hr)
		#print("Proportion target = {}".format(pt))
		#print("Proportion render = {}".format(pr))

		# Proporcion de target
		#hre = int(round(wt / pt))
		#wre = int(round(ht * pt))

		# Proporcion de render
		#hre = int(round(wt / pr))
		wre = int(round(ht * pr))

		self.render_size = (ht, wre)
		#print('Render size {}'.format(self.render_size))

	def show_render(self):
		render = self.get_img()
		self.show(render)

	def get_img(self):
		M = np.dot(self.S0, self.T0)
		M = np.dot(self.Tc, M)
		M = np.dot(self.St, M)
		M = np.dot(self.Rt, M)
		M = np.dot(np.linalg.inv(self.Tc), M)
		M = np.dot(self.Tt, M)
		#print(M)
		#print("Render size: {}".format(self.render_size))
		y,x = self.render_size
		render = 255 - cv2.warpPerspective(255 - self.img, M, (x,y),
	#		borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
			borderMode=cv2.BORDER_CONSTANT,
			#flags=cv2.INTER_CUBIC+cv2.WARP_FILL_OUTLIERS
			flags=cv2.INTER_AREA+cv2.WARP_FILL_OUTLIERS
	#		fillval=(255,255,255,0)
		)
		render = cv2.GaussianBlur(render, (5,5), self.sigma)
		return render

	def set_T0(self):
		x0, _, y0, _ = self.box

		self.T0[0, 2] = -x0 - 1
		self.T0[1, 2] = -y0 - 1

	def set_S0(self):
		assert self.target_shape != None
		ht,wt = self.target_shape
		hr,wr = self.render_shape
		s = float(ht) / float(hr)

		self.S0[0, 0] = s
		self.S0[1, 1] = s

	def set_Tc(self):
		h, w = self.render_size

		self.Tc[0, 2] = -w/2.
		self.Tc[1, 2] = -h/2.

	def set_St(self):
		self.St[0, 0] = 0.8
		self.St[1, 1] = 0.8

	def set_angle(self, angle):
		self.angle = angle
		self.Rt[0,0] = np.cos(angle)
		self.Rt[0,1] = -np.sin(angle)
		self.Rt[1,0] = np.sin(angle)
		self.Rt[1,1] = np.cos(angle)

	def prepare_shape(self, target_shape):
		self.set_shape(target_shape)
		self.bound_render()
		self.set_T0()
		self.set_S0()
		self.set_Tc()
		self.set_St()

class Target:
	def __init__(self, target_file):
		self.path = target_file
		self.load_file(target_file)
		self.size = self.img.size
		self.otsu()
		self.normalize(self.img, self.otsu)
		self.img = self.expand_hist(self.img)
		self.crop_target()
		self.extract_info(target_file)

	def extract_info(self, target):
		self.name = os.path.basename(target)
		dir_char = os.path.dirname(target)
		self.char = os.path.basename(dir_char)
		dir_font = os.path.dirname(dir_char)
		self.font = os.path.basename(dir_font)

	def load_file(self, img_file):
		#print("Load image %s" % img_file)
		self.img = cv2.imread(img_file, 0)
		assert type(self.img) == np.ndarray
		if self.img.shape[0] > 100:
			self.adjust_size()

	def adjust_size(self):
		f, c = self.img.shape
		s = 100. / float(f)
		y,x = (100, int(c*s))
		self.img = cv2.resize(self.img, (x, y), interpolation=cv2.INTER_AREA)

	def expand_hist(self, img):
		minv,maxv,minl,maxl = cv2.minMaxLoc(img)
		if(maxv-minv == 0): return img
		imgf = img.astype(np.float)
		imgf = (imgf - minv) / (maxv-minv) * 255
		img = imgf.astype(np.uint8)
		return img

	def hist_map(self, img, fmin, fmax):
		'''Coloca el valor fmin en 0 y fmax en 255.'''
		imgf = img.astype(np.float)
		imgf = (imgf - fmin) / (fmax-fmin) * 255.
		imgf[imgf > 255] = 255
		imgf[imgf < 0] = 0
		img = imgf.astype(np.uint8)
		return img

	def normalize(self, img, otsu):
		inv_otsu = 255 - otsu
		t = self.otsu_value
		img[img < t]
		img_white = img[img > t]
		img_black = img[img <= t]
		white_mean = np.mean(img_white)
		black_mean = np.mean(img_black)
		white_var = np.var(img_white)
		black_var = np.var(img_black)

		black_th = min(20, black_mean)

		img_expand = self.hist_map(self.img, black_th, white_mean)
		if DEBUG:
			print('Otsu value {}'.format(self.otsu_value))
			print("White mean {}, variance {}".format(white_mean, white_var))
			print("Black mean {}, variance {}".format(black_mean, black_var))
			cv2.imshow('expand', img_expand)
			cv2.waitKey(1)
		self.img = img_expand

	def otsu(self):
		v,th_otsu = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		self.otsu = th_otsu
		self.otsu_value = v

	def box_otsu(self, img):
		K = 1
		inv = 255 - self.otsu
		nz = inv.nonzero()
		h,w = img.shape
		x0, x1 = (np.min(nz[1]), np.max(nz[1]))
		y0, y1 = (np.min(nz[0]), np.max(nz[0]))
		x0, x1 = (np.max([x0-K, 0]), np.min([w-1, x1+K+1]))
		y0, y1 = (np.max([y0-K, 0]), np.min([h-1, y1+K+1]))
		box = (x0, x1+1, y0, y1+1)
		#return img[y0:y1+1,x0:x1+1]
		return box

	def box_sobel(self, img):
		SOBEL_K = 3
		K = 0.1

		sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=SOBEL_K).astype(np.float)
		sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=SOBEL_K).astype(np.float)

		fil, col = img.shape
		maskx = 1 + np.tile(np.arange(col), (fil,1))
		maskx_ = maskx[::,::-1]
		masky = 1 + np.tile(np.arange(fil).reshape((fil,1)), (1, col))
		masky_ = masky[::-1,::]

		distx  = np.absolute(sobelx) * np.exp(-maskx*K)
		distx_ = np.absolute(sobelx) * np.exp(-maskx_*K)
		disty  = np.absolute(sobely) * np.exp(-masky*K)
		disty_ = np.absolute(sobely) * np.exp(-masky_*K)

		_, _, _, x0 = cv2.minMaxLoc(distx)
		_, _, _, x1 = cv2.minMaxLoc(distx_)
		_, _, _, y0 = cv2.minMaxLoc(disty)
		_, _, _, y1 = cv2.minMaxLoc(disty_)

		x0y, x0x = x0
		x1y, x1x = x1
		y0y, y0x = y0
		y1y, y1x = y1
		box = (x0y, x1y+1, y0x, y1x+1)
		return box

	def crop_target(self):
		#box = self.box_sobel(self.img)
		box = self.box_otsu(self.img)
		#self.img = 255 - cv2.bitwise_and(255 - self.img, self.mask)
		#print("Target pre box = {}".format(box))
		x0, x1, y0, y1 = box
		self.target_shape = (y1-y0+2, x1-x0+2)
		f, c = self.target_shape
		mf, mc = self.img.shape
		#print("Target shape = {}".format(self.target_shape))
		x0 = int(max(0, round(x0 - c * 0.1)))
		x1 = int(min(mc-1, round(x1 + c * 0.1)))
		y0 = int(max(0, round(y0 - f * 0.1)))
		y1 = int(min(mf-1, round(y1 + f * 0.1)))
		self.box = (x0, x1, y0, y1)
		self.target_shape = (y1-y0+2, x1-x0+2)
		#print("Target box = {}".format(self.box))

	def get_shape(self):
		return self.target_shape

	def show_target(self):
		target = self.get_img()
		self.show(target)

	def get_img(self):
		x0, x1, y0, y1 = self.box
		target = self.img[y0:y1, x0:x1]
		return target

	def show(self, img = None):
		if type(img) != np.ndarray:
			img = self.img

		assert type(img) == np.ndarray
		view = pg.GraphicsView()

		l = pg.GraphicsLayout(border=(100,100,100))
		view.setCentralItem(l)
		view.show()
		ii = pg.ImageItem(img.astype(np.float).T)
		vb = l.addViewBox(lockAspect=True, invertY=True)
		vb.addItem(ii)
		QtGui.QApplication.exec_()

class Compare:
	def __init__(self, target, render):
		self.target = target
		self.render = render

	def compare(self):
		render = self.render.get_img()
		target = self.target.get_img()
		target, render = self.fill_shape(target, render)
		sim = self.compare_img(target, render)
		return (target, render, sim)

	def iterative_fit(self):
		K = 0.2
		ERR_MIN = 0.01
		target, render = (None, None)
		max_sim = 0.0
		for i in range(150):
			target, render, sim = self.compare()
			dx, dy = self.fine_tune(target, render)
			#print("Fine tune {}".format((dx, dy)))
			self.render.Tt[0, 2] += -dx * K
			self.render.Tt[1, 2] += -dy * K
			#if sim == max_sim: break
			max_sim = max(max_sim, sim)

		#diff = cv2.absdiff(target, render)
		#self.show([target, render, diff])
		#print("{:2.2f} for {}".format(max_sim*100.0, self.render.font))
		return max_sim

	def compare_img(self, target, render):
		font = self.render.font
		size = self.target.size

		diff = cv2.absdiff(target, render)
		diff_minus = 128. + (target.astype(float) - render.astype(float))/2.0
		diff_minus = self.expand_hist(diff_minus.astype(np.uint8))
		sum_diff = np.sum(diff)
		inv_target_max = np.sum(255 - target)
		#res = np.hstack((target, render))
		#diff_count = np.count_nonzero(diff)
		#sim = 1.0 - float(diff_count)/float(target.size)
		#sim = 1.0 - float(sum_diff) / float(diff.size*255)
		sim = 1.0 - float(sum_diff) / float(inv_target_max)
		#print("Similarity {:.2f}% for {}".format(sim*100, font))
		#self.show([target, render, diff])
		if 1:
			if DEBUG:
				cv2.imshow('target ', target)
				cv2.imshow('render ', render)
				cv2.imshow('diff minus', diff_minus)
			cv2.imshow('diff ', diff)
			cv2.waitKey(1)

		return sim

	def expand_hist(self, img):
		minv,maxv,minl,maxl = cv2.minMaxLoc(img)
		if(maxv-minv == 0): return img
		imgf = img.astype(np.float)
		imgf = (imgf - minv) / (maxv-minv) * 255
		img = imgf.astype(np.uint8)
		return img

	def fill_shape(self, target, render):
		tf, tc = target.shape
		rf, rc = render.shape
		f = max(tf, rf)
		c = max(tc, rc)
		nt = 255 + np.zeros((f,c), dtype=np.uint8)
		nr = 255 + np.zeros((f,c), dtype=np.uint8)
		nt[0:tf, 0:tc] = target
		nr[0:rf, 0:rc] = render
		return (nt, nr)

	def fine_tune(self, target, render):
		'''Calcula el desplazamiento que debe aplicarse sobre render para
		colocarse en target'''

		SOBEL_K = 3

		df_dx = cv2.Sobel(target, cv2.CV_64F, 1, 0, ksize=SOBEL_K).astype(np.float)
		df_dy = cv2.Sobel(target, cv2.CV_64F, 0, 1, ksize=SOBEL_K).astype(np.float)

		f = target.astype(float)
		g = render.astype(float)

		diff = (f-g) / 255.
		mx = df_dx * diff
		my = df_dy * diff

		incx = np.sum(mx) / np.sum(np.abs(mx))
		incy = np.sum(my) / np.sum(np.abs(my))

		RAND_DK = 0.5
		rand_dx = (np.random.random_sample()-0.5) * RAND_DK
		rand_dy = (np.random.random_sample()-0.5) * RAND_DK

		incx += rand_dx
		incy += rand_dy


		#print("incx = {}, incy = {}".format(incx, incy))
		(s, Sn) = self.fine_scale((mx, -my), render.shape)

		RAND_SK = 0.005
		rand_s = (np.random.random_sample() - 0.5) * RAND_SK

		s += rand_s
		self.render.St[0,0] *= s
		self.render.St[1,1] *= s

		dsigma = 0
		dsigma = self.fine_smooth(target, render, diff)

		#self.fine_rotation(target, render, diff)
		#pgplot.add_imgs(l, [diff, mx, my])
		#return (incx, incy, s, mx, my, Sn)
		#print("{:2.3f}\t{:2.3f}\t{:2.3f}\t{:2.3f}".format(incx, incy, s, dsigma))
		return (incx, incy)

	def fine_scale(self, m, shape):
		K = 1.0
		N, M = shape
		m = m/np.max(np.abs(m))
		PMx, PMy = (-m[0], -m[1])
		#PMx, PMy = (m[0], m[1])

		OC = np.array([(M-1)/2.,(N-1)/2.])
		OP = np.zeros([N, M, 2])
		for i in range(N):
			for j in range(M):
				OP[i,j] = [j, N-i-1]

		CP = OP - OC

		PM = np.array([PMx, PMy])
		PM = PM.reshape([2,N*M]).T.reshape([N,M,2])

		CM = CP + PM
		CPn = np.sqrt(CP[:,:,0]**2 + CP[:,:,1]**2)
		CPn[(N-1)/2,(M-1)/2] = 1

		CPb = np.zeros([N, M, 2])
		CPb[:,:,0] = CP[:,:,0] / CPn
		CPb[:,:,1] = CP[:,:,1] / CPn

		CQ = CM * CPb
		CQn = CQ[:,:,0] + CQ[:,:,1]
		Sn = CQn/CPn
		Sn[(N-1)/2-(N-1)/10:(N-1)/2+(N-1)/10,(M-1)/2-(M-1)/10:(M-1)/2+(M-1)/10] = 1
		s = np.sum(Sn)/(N*M)
		s = 1+((s-1)*K)
		#print("Mejor escalado = {}".format(s))
		#self.show([PMx, PMy, CPb[:,:,0], CPb[:,:,1], CQn, Sn])

		return (s, Sn)

	def fine_smooth(self, target_img, render_img, diff_img):
		SIGMA_INC = 0.5
		K = 0.03
		SIGMA_DIFF = 0.4

		sigma2 = self.render.sigma + SIGMA_INC
		render_smooth = cv2.GaussianBlur(render_img.astype(float), (5,5), sigma2)

		diff_smooth = (render_smooth - render_img)/255.
		h = diff_smooth * diff_img
		diff_sigma = np.sum(h) / np.sum(np.abs(h))
		diff_sigma = np.nan_to_num(diff_sigma)
		dsigma = (diff_sigma-SIGMA_DIFF) * K
		#self.show([target_img, diff_img, render_img, render_smooth, diff_smooth,h])

		RAND_SIGMA = 0.01
		rand_sigma = (np.random.random_sample()-0.5) * RAND_SIGMA
		dsigma += rand_sigma

		self.render.sigma += dsigma
		#print("dsigma = {}".format(dsigma))
		#print("Render sigma = {}".format(self.render.sigma))

		return dsigma

	def fine_rotation(self, target_img, render_img, diff_img):
		angle0 = self.render.angle
		INC_ANGLE = 0.01
		K = 0.001

		render0 = render_img.astype(float)
		self.render.set_angle(angle0 + INC_ANGLE)
		render1 = self.render.get_img().astype(float)
		self.render.set_angle(angle0)
		target_img, render1_filled = self.fill_shape(target_img, render1)

		f = target_img.astype(float)

		diff_angle = (render1_filled - render0) / INC_ANGLE
		diff_target = f - render0
		field = diff_target / (diff_angle + 1)

		n,m = field.shape
		inc_angle = np.sum(field) / np.sum(np.abs(field)) * K
		#print(inc_angle)
		self.render.set_angle(angle0 + inc_angle)
		#self.show([target_img, diff_target, render0, render1_filled, diff_angle,field])
		cv2.imshow('rot ', self.expand_hist(field))
		#cv2.waitKey(500)
		cv2.waitKey(1)

	def show(self, img_list):
		view = pg.GraphicsView()
		l = pg.GraphicsLayout(border=(100,100,100))
		view.setCentralItem(l)
		view.show()
		for img in img_list:
			ii = pg.ImageItem(img.astype(np.float).T)
			vb = l.addViewBox(lockAspect=True, invertY=True)
			vb.addItem(ii)
		QtGui.QApplication.exec_()

class Recognize:
	def __init__(self, target, char):
		self.target = target
		self.char = char

	def natural_sort(self, l):
		convert = lambda text: int(text) if text.isdigit() else text.lower()
		alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
		return sorted(l, key = alphanum_key)

	def recognize_font(self):
		if DEBUG:print("Idendify {}".format(self.target.path))
		render_names = self.natural_sort(os.listdir(RENDER_DIR))
		best_font = None
		max_sim = 0.0
		results = []
		for font in render_names:
			r = Render(font, self.char)
			r.prepare_shape(self.target.get_shape())
			c = Compare(self.target, r)
			sim = c.iterative_fit()
			if sim > max_sim:
				max_sim = sim
				best_font = font
			results.append((font, sim))
			if DEBUG:print("{:2.2f} for {}".format(sim*100.0, font))
		if DEBUG:print("Best font is {} at {:2.2f}".format(best_font, max_sim*100.0))
		return (best_font, results)

class Analize:
	def __init__(self):
		self.target_names = self.find_targets(TARGET_DIR)

	def natural_sort(self, l):
		convert = lambda text: int(text) if text.isdigit() else text.lower()
		alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
		return sorted(l, key = alphanum_key)

	def find_targets(self, dir_name):
		result = []
		for root, dirs, files in os.walk(dir_name):
			for name in files:
				result.append(os.path.join(root, name))
		return self.natural_sort(result)

	def extract_info(self, target):
		name = os.path.basename(target)
		dir_char = os.path.dirname(target)
		char = os.path.basename(dir_char)
		dir_font = os.path.dirname(dir_char)
		font = os.path.basename(dir_font)
		return (font, char, name)

	def compare(self, target, font):
		r = Render(font, target.char)
		r.prepare_shape(target.get_shape())
		c = Compare(target, r)
		sim = c.iterative_fit()
		return sim

	def print_cell(self, value):
		sys.stdout.write('{:2.1f}%\t'.format(value*100.))
		sys.stdout.flush()

	def print_header(self):
		fonts = self.natural_sort(os.listdir(RENDER_DIR))
		for font in fonts:
			sys.stdout.write('{:.7}\t'.format(font))
		print('f  s  Target file')

	def family(self, font_name):
		p = re.compile('[^0-9]+')
		m = p.match(font_name)
		return m.group(0)

	def print_table(self):
		fonts = self.natural_sort(os.listdir(RENDER_DIR))
		self.print_header()
		font_err = 0
		fam_err = 0
		for target_file in self.target_names:
			t = Target(target_file)
			target_family = self.family(t.font)
			bes_font = None
			max_s = 0.0
			for font in fonts:
				s = self.compare(t, font)
				self.print_cell(s)
				if(s > max_s):
					best_font = font
					max_s = s
			best_family = self.family(best_font)
			if best_family == target_family:
				sys.stdout.write('+f ')
			else:
				sys.stdout.write('-f ')
				fam_err+=1


			if best_font == t.font:
				sys.stdout.write('+s ')
			else:
				sys.stdout.write('-s ')
				font_err+=1

			print('{}'.format(target_file))

		print("{}/{} family errors.".format(fam_err, len(self.target_names)))
		print("{}/{} font and size errors.".format(font_err, len(self.target_names)))

	def all(self):
		errors = 0
		self.table = []
		for target_file in self.target_names:
			font, char, name = self.extract_info(target_file)
			#print("Font {}, char {}, file {}".format(
			#	font, char, name))
			t = Target(target_file)
			best, results = Recognize(t, char).recognize_font()
			self.table.append((target_file, results))

def main():
	app = QtGui.QApplication([])
	#target_file = 'data/target/cmr10/a/IMG_0044.png'
	#target_file = 'data/target/lucida-bright-regular/a/a1-jahne.png'
	#target_file = 'data/target/cmr10/b/1.png'
	#target_file = 'data/target/cmr10/a/big1.png'
	if len(sys.argv) == 2:
		target_file = sys.argv[1]
		t = Target(target_file)
		#t.show_target()
		#r = Render('lucida-bright-regular', 'a')
		#r = Render('cmr10', 'b')
		#r.prepare_shape(t.get_shape())
		#r.show_render()
		#c = Compare(t, r)
		#c.iterative_fit()
		Recognize(t, t.char).recognize_font()
	else:
		Analize().print_table()


if __name__ == '__main__':
	if len(sys.argv) == 2: DEBUG = True
	main()

