from PIL import Image, ImageDraw
import os
import cv2


def make_images(data_dir, is_right, write=False):
  label = 'r' if is_right else 'l'
  images = []
  prefix = 'right: ' if is_right else 'left: '
  color = (191, 132, 71) if is_right else (71, 71, 191)

  im = cv2.imread(os.path.join(data_dir, 'disp_' + label +'_init.png'))
  cv2.putText(im, prefix + "random initialization", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
  images += [im]*10

  for i in range(3):
    for j in range(5):
      name = 'disp_' + label + '_' + str(i) + '_' + str(j) + '.png'
      path = os.path.join(data_dir, name)
      im = cv2.imread(path)
      cv2.putText(im, prefix + "iter " + str(i), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
      images.append(im)

  im = cv2.imread(os.path.join(data_dir, 'disp_' + label +'_iterend.png'))
  cv2.putText(im, prefix + "finish main process", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
  images += [im]*10

  im = cv2.imread(os.path.join(data_dir, 'disp_' + label +'_lrconsistency.png'))
  cv2.putText(im, prefix + "left right consistency", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
  images += [im]*10

  im = cv2.imread(os.path.join(data_dir, 'disp_'+ label +'_fillholenn.png'))
  cv2.putText(im, prefix + "hole filling", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
  images += [im]*10

  im = cv2.imread(os.path.join(data_dir, 'disp_'+ label +'_filled.png'))
  cv2.putText(im, prefix + "weighted median filter", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
  images += [im]*10

  im = cv2.imread(os.path.join(data_dir, 'disp_'+ label +'_filled.png'))
  cv2.putText(im, prefix + "final output", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
  images += [im]*10
  
  if write:
    for i, im in enumerate(images):
      name = 'gif_' + label +  '_' + str(i).zfill(2) + '.png'
      path = os.path.join(data_dir, name)
      cv2.imwrite(path, im)
  return images

def make_gif(root_dir, prefix, ext):
  images = []

  im_files = sorted([x for x in os.listdir(root_dir) if x.startswith(prefix) and x.endswith(ext)])

  for im_file in im_files:
    path = os.path.join(root_dir, im_file)
    im = Image.open(path)
    images.append(im)

  images[0].save(prefix + '.gif',
                save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)

if __name__ == "__main__":
  data_dir = '../win_build/'
  left_images = make_images(data_dir, False)
  right_images = make_images(data_dir, True)
  for i, (l, r) in enumerate(zip(left_images, right_images)):
    combined = cv2.hconcat([l, r])
    name = 'gif_' + str(i).zfill(2) + '.png'
    path = os.path.join(data_dir, name)
    cv2.imwrite(path, combined)
  make_gif(data_dir, 'gif', 'png')
  #make_gif(data_dir, 'gif_r', 'png')