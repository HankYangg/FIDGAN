import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
import os
# import matplotlib.pyplot as plt
from util import util
import shutil
fake_path = '/root/AFS/pairwise_xray_augmentation/fake_image/'

if __name__ == '__main__':
	opt = TrainOptions().parse()
	netDtype = opt.netD
	data_loader = CreateDataLoader(opt) # init.py
	dataset = data_loader.load_data()
	dataset_size = len(data_loader)
	print('#training images = %d' % dataset_size)
	save_dir = os.path.join(opt.checkpoints_dir, opt.name)

	model = create_model(opt)
	model.setup(opt)
	total_steps = 0
	
	f1 = open(opt.name+'_G_GAN_loss.txt','w')
	f2 = open(opt.name+'_G_L1_loss.txt','w')
	f3 = open(opt.name+'_D_real_loss.txt','w')
	f4 = open(opt.name+'_D_fake_loss.txt','w')
	f5 = open(opt.name+'_FID_loss.txt','w')
	
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1): # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
		epoch_start_time = time.time() # timer for entire epoch
		iter_data_time = time.time() # timer for data loading per iteration
		epoch_iter = 0 # the number of training iterations in current epoch, reset to 0 every epoch

		for i, data in enumerate(dataset): # inner loop within one epoch
			iter_start_time = time.time() # timer for computation per iteration
			if total_steps % opt.print_freq == 0:
				t_data = iter_start_time - iter_data_time
			
			total_steps += opt.batch_size
			epoch_iter += opt.batch_size
			model.set_input(data) # unpack data from dataset and apply preprocessing
			model.optimize_parameters() # calculate loss functions, get gradients, update network weights
			
			#############################################
			#visuals = model.get_current_visuals()
			#print('visuals:   ', visuals)
			#for label, im_data in visuals.items():
				#im = util.tensor2im(im_data)
				#print('yes, inin')
				#print('im:  ', im)
				#img_name = 'image_%s_%s.png' % (epoch, label)
				#img_loc = os.path.join(save_dir, img_name)
				#util.save_image(im, img_loc)
			
			
			#############################################
			if total_steps % opt.print_freq == 0: # print training losses and save logging information to the disk
				losses = model.get_current_losses()
				############################hank
				print(losses)
		
				f1.write(str(losses['G_GAN'])+'\n')
				f2.write(str(losses['G_L1'])+'\n')
				f3.write(str(losses['D_real'])+'\n')
				f4.write(str(losses['D_fake'])+'\n')
				f4.write(str(losses['FID'])+'\n')
				############################
				t = (time.time() - iter_start_time) / opt.batch_size

			if total_steps % opt.save_latest_freq == 0: # cache our latest model every <save_latest_freq> iterations
				print('saving the latest model (epoch %d, total_steps %d)' %
							(epoch, total_steps))
				model.save_networks('latest')

			iter_data_time = time.time()
		if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
			print('saving the model at the end of epoch %d, iters %d' %
						(epoch, total_steps))
			model.save_networks('latest')
			model.save_networks(epoch)
			
			visuals = model.get_current_visuals()
			#print('visuals:   ', visuals)
			for label, im_data in visuals.items():
				im = util.tensor2im(im_data)
				#print('im:  ', im)
				img_name = 'image_%s_%s.png' % (epoch, label)
				img_loc = os.path.join(save_dir, img_name)
				util.save_image(im, img_loc)

		print('End of epoch %d / %d \t Time Taken: %d sec' %
					(epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
		shutil.rmtree(fake_path)
		if not os.path.exists(fake_path):
			os.mkdir(fake_path)
		model.update_learning_rate()
	f1.close()
	f2.close()
	f3.close()
	f4.close()