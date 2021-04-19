import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util import util

if __name__ == '__main__':
	opt = TestOptions().parse()
	# hard-code some parameters for test
	opt.num_threads = 1  # test code only supports num_threads = 1
	opt.batch_size = 1  # test code only supports batch_size = 1
	opt.serial_batches = True  # no shuffle
	opt.no_flip = True  # no flip
	opt.display_id = -1  # no visdom display
	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	model = create_model(opt)
	model.setup(opt)
	
	save_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.name+"_fake_images")
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	# pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
	if opt.eval:
		model.eval()
	for epoch in range(10):

		for i, data in enumerate(dataset):
			if i + epoch * len(dataset) >= opt.num_test:
				break
			model.set_input(data)
			model.test()
			visuals = model.get_current_visuals()
			img_path = model.get_image_paths()
			for label, im_data in visuals.items():
				if label == 'fake_B':
					im = util.tensor2im(im_data)
					#img_name = data['A_paths'][0][-16:-4]+'_2_%s.png'%(epoch+1)
					img_name = data['A_paths'][0][-30:-4]+'.png' # ori:30, flip:32
					img_loc = os.path.join(save_dir, img_name)
					util.save_image(im, img_loc)

			if i % 50 == 0:
				print('processing (%04d)-th image... %s' % (i, img_path))
