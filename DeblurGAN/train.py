import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM




def write_yolo_loss(model,step):
	### Tong: This function writes step and yolo loss 
	yolo_loss=model.get_yolo_loss() ### Tong Lin: write yolo loss to txt
	fileID=open("train/trial/yolo_loss.txt","a")
	fileID.write(str(step)+" "+str(yolo_loss)+"\n")
	fileID.close() 

def train(opt, data_loader, model, visualizer):
	dataset = data_loader.load_data()
	#print (dataset)
	dataset_size = len(data_loader)
	print('#training images = %d' % dataset_size)
	total_steps = 0
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		epoch_start_time = time.time()
		epoch_iter = 0
		for i, data in enumerate(dataset):
			iter_start_time = time.time()
			#print (opt.batchSize)
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize
			model.set_input(data)
			model.optimize_parameters()  ### Tong Lin: training step: forward and backward 
			if total_steps % opt.display_freq == 0:
				results = model.get_current_visuals()
				psnrMetric = PSNR(results['Restored_Train'],results['Sharp_Train'])
				print('PSNR on Train = %f' %
					  (psnrMetric))
				visualizer.display_current_results(results,epoch)
			if total_steps % opt.print_freq == 0:
				errors = model.get_current_errors()
				t = (time.time() - iter_start_time) / opt.batchSize
				visualizer.print_current_errors(epoch, epoch_iter, errors, t)
				if opt.display_id > 0:
					visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
				write_yolo_loss(model,total_steps) ## Tong Lin: This function writes yolo_loss every 100 steps
				print ("total_steps:",total_steps,"Error:",errors) ## Tong Lin: print steps and errors every 100 steps
			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' %
					  (epoch, total_steps))
				model.save('latest'+str(total_steps))  ### Tong Lin: record model every 5000 steps 
		print ("Epoch:",epoch)
		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' %
				  (epoch, total_steps))
			model.save('latest_epoch')
			model.save(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' %
			  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

		if epoch > opt.niter:
			model.update_learning_rate()
			
opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
model = create_model(opt)
visualizer = Visualizer(opt)
train(opt, data_loader, model, visualizer)
