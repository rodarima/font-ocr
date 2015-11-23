from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
FONT_DIR = "data/font"
FONT_SIZE = 500
RENDER_DIR = "data/render"
RENDER_STR = "{}.png"
TARGET_DIR = "data/target"

# Genera una bd de imagenes con la letras del alfabeto y las almacena en un
# directorio con el nombre de la fuente y el de la letra.
# Adicionalmente crea las carpetas adecuadas en TARGET_DIR con el nombre de cada
# fuente, de modo que se puedan colocar las muestras clasificadas.

def crop_char(img):
	inv = 255 - img
	nz = inv.nonzero()
	x0, x1 = (np.min(nz[1]), np.max(nz[1]))
	y0, y1 = (np.min(nz[0]), np.max(nz[0]))
	return img[y0:y1+1,x0:x1+1]

def draw_char(font, size, char):
	big_size = (size[0]*3, size[1]*3)
	img = Image.new('L', big_size, 255)
	d = ImageDraw.Draw(img)
	d.text(size, char, font=font, fill=0)
	real = np.array(img.convert('RGB'))
#	real = crop_char(real)
	return real

def generate_char(font, char):
	size = font.getsize(char)
	#print("Size for {} is {}".format(char, size))
	img = draw_char(font, size, char)
	return img

def render_file(font_file, char):
	font_name = os.path.splitext(font_file)[0]
	render = RENDER_STR.format(char)
	render_path = os.path.join(RENDER_DIR, font_name)
	if not os.access(render_path, os.R_OK):
		os.makedirs(render_path)
	render_filename = os.path.join(render_path, render)
	return render_filename

def save_char(font_file, char, img):
	name = render_file(font_file, char)
	#print("Saving {}".format(name))
	cv2.imwrite(name, img)

def generate_font(name):
	font_file = os.path.join(FONT_DIR, name)
	font = ImageFont.truetype(font_file, FONT_SIZE)
	for i in range(len(ALPHABET)):
		char = ALPHABET[i]
		char_file = render_file(name, char)
		if not os.access(char_file, os.R_OK):
			img = generate_char(font, char)
			save_char(name, char, img)
		#print("\b{:2.2f}    ".format(float(i)/float(len(ALPHABET))*100))

def target_dirs(target_dir):
	for char in ALPHABET:
		char_dir = os.path.join(target_dir, char)
		if not os.access(char_dir, os.R_OK):
			os.mkdir(char_dir)

def generate_target(font_file):
	font_name = os.path.splitext(font_file)[0]
	target_dir = os.path.join(TARGET_DIR, font_name)
	#if os.access(target_dir, os.R_OK): return
	if not os.access(target_dir, os.R_OK):
		print("New target {}".format(target_dir))
		os.makedirs(target_dir)
	target_dirs(target_dir)

def update_db():
	font_names = os.listdir(FONT_DIR)
	for i in range(len(font_names)):
		name = font_names[i]
		generate_font(name)
		generate_target(name)
		print("Done {:2.2f}".format(i*100/len(font_names)))

update_db()

