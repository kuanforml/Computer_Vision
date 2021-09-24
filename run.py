#!/usr/bin/env python
import torch
import getopt
import math
import numpy as np
import os
import PIL
import PIL.Image
from PIL import Image
import sys
import argparse
import torchvision
from torchvision import transforms
import cv2
#import cv2.ximgproc as xip
from dataclasses import dataclass
import matplotlib.pyplot as plt
import softsplat

try:
	from .correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'default' # 'default', or 'chairs-things'
arguments_strFirst = './0_center_frame/7/input/frame10.png'
arguments_strSecond = './0_center_frame/7/input/frame11.png'
arguments_strOut = './out.flo'

#for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
#	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
#	if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument # path to the first frame
#	if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument # path to the second frame
#	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

backwarp_tenGrid = {}
backwarp_tenPartial = {}
BLUR_OCC = 3
NEIGH_PIXELS = 8
def readFlowFile(file):
    '''
    credit: this function code is obtained from: https://github.com/Johswald/flow-code-python
    '''
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end
	print("hi")
	if str(tenFlow.shape) not in backwarp_tenPartial:
		backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
	tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)

	tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

	tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

	return tenOutput[:, :-1, :, :] * tenMask
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tenInput):
				tenOne = self.netOne(tenInput)
				tenTwo = self.netTwo(tenOne)
				tenThr = self.netThr(tenTwo)
				tenFou = self.netFou(tenThr)
				tenFiv = self.netFiv(tenFou)
				tenSix = self.netSix(tenFiv)

				return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
			# end
		# end

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

				if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
				)
			# end

			def forward(self, tenFirst, tenSecond, objPrevious):
				tenFlow = None
				tenFeat = None

				if objPrevious is None:
					tenFlow = None
					tenFeat = None

					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFirst, tenSecond=tenSecond), negative_slope=0.1, inplace=False)

					tenFeat = torch.cat([ tenVolume ], 1)

				elif objPrevious is not None:
					tenFlow = self.netUpflow(objPrevious['tenFlow'])
					tenFeat = self.netUpfeat(objPrevious['tenFeat'])

					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFirst, tenSecond=backwarp(tenInput=tenSecond, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

					tenFeat = torch.cat([ tenVolume, tenFirst, tenFlow, tenFeat ], 1)

				# end

				tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

				tenFlow = self.netSix(tenFeat)

				return {
					'tenFlow': tenFlow,
					'tenFeat': tenFeat
				}
			# end
		# end

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()

				self.netMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)
			# end

			def forward(self, tenInput):
				return self.netMain(tenInput)
			# end
		# end

		self.netExtractor = Extractor()

		self.netTwo = Decoder(2)
		self.netThr = Decoder(3)
		self.netFou = Decoder(4)
		self.netFiv = Decoder(5)
		self.netSix = Decoder(6)

		self.netRefiner = Refiner()

		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + arguments_strModel + '.pytorch', file_name='pwc-' + arguments_strModel).items() })
	# end

	def forward(self, tenFirst, tenSecond):
		tenFirst = self.netExtractor(tenFirst)
		tenSecond = self.netExtractor(tenSecond)

		objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
		objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
		objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
		objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)
		objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)

		return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])
	# end
# end

netNetwork = None

##########################################################

def estimate(tenFirst, tenSecond):
	global netNetwork

	if netNetwork is None:
		netNetwork = Network().cuda().eval()
	# end

	assert(tenFirst.shape[1] == tenSecond.shape[1])
	assert(tenFirst.shape[2] == tenSecond.shape[2])

	intWidth = tenFirst.shape[2]
	intHeight = tenFirst.shape[1]

	#assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	#assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
	tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

	tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tenFlow = 20.0 * torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
	#print(tenFlow)
	#print(tenFlow.type) 
	return tenFlow[:, :, :, :].cpu()
# end

##########################################################
##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

##########################################################

def read_flo(strFile):
    with open(strFile, 'rb') as objFile:
        strFlow = objFile.read()
    # end

    assert(numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=1, offset=0) == 202021.25)

    intWidth = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=4)[0]
    intHeight = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=8)[0]

    return numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=intHeight * intWidth * 2, offset=12).reshape([ intHeight, intWidth, 2 ])
# end

##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# end

##########################################################

#import model
#import softsplat
#def save_flow(flow, output_file):
#    objOutput = open(output_file, 'wb')
#    flow = flow[0, :, :, :]
#    np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objOutput)
#    np.array([ flow.shape[2], flow.shape[1] ], np.int32).tofile(objOutput)
#    np.array(flow.cpu().numpy().transpose(1, 2, 0), np.float32).tofile(objOutput)
#
#    objOutput.close()
#def get_optical_flow(Img0, Img1):
#    tenFirst = torch.FloatTensor(np.ascontiguousarray(Img0[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
#    tenSecond = torch.FloatTensor(np.ascontiguousarray(Img1[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
#    tenOutput = estimate(tenFirst, tenSecond)
#    #print(tenOutput[:, 395,360])
#    return tenOutput
#
#def backward_warping(img, flow):
#    _, _, H, W = img.size()
#    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
#    gridX = torch.tensor(gridX, requires_grad=False).cuda()
#    gridY = torch.tensor(gridY, requires_grad=False).cuda()
#    #print(flow.shape)
#    u = flow[:,0,:,:]
#    v = flow[:,1,:,:]
#
#    #u = flow[0,:,:]
#    #v = flow[1,:,:]
#    x = gridX.unsqueeze(0).expand_as(u).float()+u
#    y = gridY.unsqueeze(0).expand_as(v).float()+v
#    normx = 2*(x/W-0.5)
#    normy = 2*(y/H-0.5)
#    grid = torch.stack((normx,normy), dim=3)
#    warped = torch.nn.functional.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
#    #print(warped.shape)
#    return warped
#
#def backward_warping_1(img, flow):
#    B, C, H, W = img.size()
#        # mesh grid 
#    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
#    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
#    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
#    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
#    grid = torch.cat((xx,yy),1).float()
#
#    if img.is_cuda:
#        grid = grid.cuda()
#    vgrid = Variable(grid) + flow
#
#    # scale grid to [-1,1] 
#    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
#    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0
#
#    vgrid = vgrid.permute(0,2,3,1)        
#    output = F.grid_sample(img, vgrid, align_corners=True)
#    mask = torch.autograd.Variable(torch.ones(img.size())).cuda()
#    mask = F.grid_sample(mask, vgrid, align_corners=True)
#
#        
#    mask[mask<0.9999] = 0
#    mask[mask>0] = 1
#        
#    return output*mask
#
#
#def length_sq(x):
#    return torch.sum(torch.square(x), 0)
#
#def occlusion(flow_fw, flow_bw, alpha_1=0.01, alpha_2=0.5):
#    # need 4D tensor
#    flow_bw_warped = backward_warping(flow_bw, flow_fw) # F_0_1_back
#    flow_fw_warped = backward_warping(flow_fw, flow_bw) # F_1_0_back
#    flow_fw_3d = flow_fw[0, :, :, :]
#    flow_bw_3d = flow_bw[0, :, :, :]
#    flow_bw_warped_3d = flow_bw_warped[0, :, :, :]
#    flow_fw_warped_3d = flow_fw_warped[0, :, :, :]
#    diff_0_1_file = './diff_0_1.flo'
#    diff_1_0_file = './diff_1_0.flo'
#    save_flow(flow_fw+flow_bw_warped, diff_0_1_file)
#    save_flow(flow_bw+flow_fw_warped, diff_1_0_file)
#    flow_diff_fw = flow_fw_3d + flow_bw_warped_3d
#    flow_diff_bw = flow_bw_3d + flow_fw_warped_3d
#    
#    mag_sq_fw = length_sq(flow_fw_3d) - length_sq(flow_bw_warped_3d)
#    mag_sq_bw = length_sq(flow_bw_3d) - length_sq(flow_fw_warped_3d)
#    #print(mag_sq_fw.shape)
#    occ_thresh_fw = alpha_1 * mag_sq_fw + alpha_2
#    occ_thresh_bw = alpha_1 * mag_sq_bw + alpha_2
#    occ_fw = length_sq(flow_diff_fw) < occ_thresh_fw
#    occ_fw = occ_fw.type(torch.FloatTensor)
#    occ_bw = length_sq(flow_diff_bw) < occ_thresh_bw
#    occ_bw = occ_bw.type(torch.FloatTensor)
#    
#    return occ_fw, occ_bw
#
#def slomo_vision_map(I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0):
#    ArbTimeFlowIntrp = model.UNet(20, 5)
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    ArbTimeFlowIntrp.to(device)
#    for param in ArbTimeFlowIntrp.parameters():
#        param.requires_grad = False
#    checkpoint = 'SuperSloMo.ckpt'
#    dict1 = torch.load(checkpoint, map_location='cpu')
#    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
#
#    intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
#    F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
#    F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
#    V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
#    V_t_1   = 1 - V_t_0
#    return V_t_0, V_t_1, F_t_0_f, F_t_1_f
#def slomo_flow(I0, I1):
#    transform = transforms.ToTensor()
#    flowComp = model.UNet(6, 4)
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    flowComp.to(device)
#    I0 = transform(I0).unsqueeze(0).cuda()
#    I1 = transform(I1).unsqueeze(0).cuda()
#    for param in flowComp.parameters():
#        param.requires_grad = False
#    checkpoint = 'SuperSloMo.ckpt'
#    dict1 = torch.load(checkpoint, map_location='cpu')
#    flowComp.load_state_dict(dict1['state_dictFC'])
#    flowOut = flowComp(torch.cat((I0, I1), dim=1))
#    F_0_1 = flowOut[:,:2,:,:]
#    F_1_0 = flowOut[:,2:,:,:]
#    return F_0_1, F_1_0
#
#def backwarp(tenInput, tenFlow):
#    backwarp_tenGrid = {}
#    if str(tenFlow.shape) not in backwarp_tenGrid:
#      tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
#      tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
# 
#      backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
#     # end
#
#      tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
#
#      return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# # end
#def interp_forward(I0, I1, t=0.5):
#    Img0 = Image.open(I0)
#    Img1 = Image.open(I1)
#    img0 = np.array(Img0)
#    img1 = np.array(Img1)
#    transform = transforms.ToTensor() 
#    tenfirst = torch.FloatTensor(np.ascontiguousarray(img0[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).cuda()
#    print(tenfirst.shape)
#    tensecond = torch.FloatTensor(np.ascontiguousarray(img1[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))).cuda()
#    flow_0_1 = get_optical_flow(img0, img1)
#    flow_1_0 = get_optical_flow(img1, img0)
#    img0 = transform(Img0).unsqueeze(0).cuda()
#    img1 = transform(Img1).unsqueeze(0).cuda()
#    _, H, W = tensecond.size()
#    flow_0_1 = flow_0_1.unsqueeze(0).cuda()
#    flow_1_0 = flow_1_0.unsqueeze(0).cuda()
#    res0 = backward_warping(img0, flow_1_0)
#    res1 = backward_warping(img1, flow_0_1)
#    print(tenfirst.shape)
#    print(res1.shape)
#    tenMetric = torch.nn.functional.l1_loss(input=tenfirst, target=res1,reduction='none').mean(1, True)
#    tenSoftmax = softsplat.FunctionSoftsplat(tenInput=tenfirst, tenFlow=flow_0_1 * t, tenMetric=-20.0 * tenMetric, strType='softmax')
#    tenSoftmax = tenSoftmax[:,[2,1,0],:,:]
#    tenSoftmax_0 = tenSoftmax[0,:,0:H,0:W].squeeze(0).cpu()
#    
#    tenSoftmax_0 = transforms.functional.to_pil_image(tenSoftmax_0)
#    tenSoftmax_0.save('test_1.png')
#
#    tenMetric_1 = torch.nn.functional.l1_loss(input=tensecond, target=res0,reduction='none').mean(1, True)
#    tenSoftmax_1 = softsplat.FunctionSoftsplat(tenInput=tensecond, tenFlow=flow_1_0 * t, tenMetric=-20.0 * tenMetric_1, strType='softmax')
#    
#    tenSoftmax_1 = tenSoftmax_1[:,[2,1,0],:,:]
#    tenSoftmax_2 = tenSoftmax_1[0,:,0:H,0:W].squeeze(0).cpu()
#    tenSoftmax_2 = transforms.functional.to_pil_image(tenSoftmax_2)
#    tenSoftmax_2.save('test_0.png')
#
#    flow_t_0 = -(1-t)*t*flow_0_1 + t*t*flow_1_0
#    flow_t_1 = (1-t)*(1-t)*flow_0_1 - t*(1-t)*flow_1_0
#    res_0 = backward_warping(img0, flow_t_0)
#    res_1 = backward_warping(img1, flow_t_1)
#    V_t_0, V_t_1, refined_flow_t_0, refined_flow_t_1 = slomo_vision_map(img0, img1, flow_0_1, flow_1_0, flow_t_1, flow_t_0, res_0, res_1)
#    V_t_0_pic = V_t_0[0, 0, :, :]
#    V_t_1_pic = V_t_1[0, 0, :, :]
#    V_t_0 = torch.round(V_t_0)
#    V_t_1 = torch.round(V_t_1)
#    res = ((1-t)*V_t_0*tenSoftmax+t*V_t_1*tenSoftmax_1)/((1-t)*V_t_0+t*V_t_1)
#      #res = (0.5*res_0+0.5*res_1)/1
#    res = res[0,:,0:H,0:W].squeeze(0).cpu()
#
#    output = transforms.functional.to_pil_image(res)
#      #output = TP(res)
#    output.save('./im_interp1.png')
#def range_map(flow_0_1, flow_1_0, t=0.5):
#    # 4D Tensor
#    transform = transforms.ToTensor()
#    _, _, H, W = flow_0_1.size()
#    one_map = np.ones((1, H, W))
#    one_map = transform(one_map).cuda().type(torch.FloatTensor)
#    one_map_4d = one_map.unsqueeze(0).permute(2,0,3,1).cuda()
#    
#    one_map_1_0 = backward_warping(one_map_4d, flow_1_0)
#    one_map_0_1 = backward_warping(one_map_4d, flow_0_1)
#    print(one_map.shape)
#    print(one_map_0_1.shape)
#    tenMetric = torch.nn.functional.l1_loss(input=one_map.permute(1,2,0).cuda(), target=one_map_0_1,reduction='none').mean(1, True)
#    tenSoftmax = softsplat.FunctionSoftsplat(tenInput= one_map.permute(1,2,0).cuda(), tenFlow=flow_0_1 * t, tenMetric=-20.0 * tenMetric, strType='softmax')
#    tenSoftmax = tenSoftmax[:,[2,1,0],:,:]
#    tenSoftmax_0 = tenSoftmax[0,:,0:H,0:W].squeeze(0)
#    return tenSoftmax_0
#    
#
#
#    
#    
#
#def interp_frame(I0, I1, t=0.5, thre=0.55):
#    
#    Img0 = cv2.cvtColor(cv2.imread(I0), cv2.COLOR_BGR2RGB)
#    Img1 = cv2.cvtColor(cv2.imread(I1), cv2.COLOR_BGR2RGB)
#    H, W, _ = Img1.shape
#    H_,W_ = int(np.ceil(H/32)*32),int(np.ceil(W/32)*32)
#    Img0 = cv2.copyMakeBorder(Img0, 0, H_-H, 0, W_-W, cv2.BORDER_REPLICATE)
#    Img1 = cv2.copyMakeBorder(Img1, 0, H_-H, 0, W_-W, cv2.BORDER_REPLICATE)
#    Img0 = Image.fromarray(Img0)
#    Img1 = Image.fromarray(Img1)
#
#    #Img0 = Image.open(I0)
#    #Img1 = Image.open(I1)
#    img0 = np.array(Img0)
#    img1 = np.array(Img1)
#    transform = transforms.ToTensor()
#    flow_0_1 = get_optical_flow(img0, img1).numpy()
#    flow_1_0 = get_optical_flow(img1, img0).numpy()
#    
#    flow_0_1 = transform(flow_0_1[0,:,:,:])
#    flow_1_0 = transform(flow_1_0[0,:,:,:])
#    
#    #flow_0_1, flow_1_0 = slomo_flow(Img0, Img1)
#    
#    
#    flow_0_1_file = './flow_0_1.flo'
#    flow_0_1_back_file = './flow_0_1_back.flo'
#    flow_1_0_file = './flow_1_0.flo'
#    flow_1_0_back_file = './flow_1_0_back.flo'
#
#    with torch.no_grad():
#      img0 = transform(Img0).unsqueeze(0).cuda()
#      img1 = transform(Img1).unsqueeze(0).cuda()
#      
#      
#      #print(img0.shape)
#      
#      flow_0_1 = flow_0_1.unsqueeze(0).permute(0,2,3,1).cuda()
#      flow_1_0 = flow_1_0.unsqueeze(0).permute(0,2,3,1).cuda()
#      '''
#      range_map_0 = range_map(flow_0_1, flow_1_0)
#      print(range_map_0.shape)
#      range_map_0_pic = range_map_0[0,0,:,:].cpu()
#      k = transforms.functional.to_pil_image(range_map_0_pic)
#      k.save('range_map_0.png')
#      print('fuck')
#      '''
#      #print(range_map_0.shape)
#      #print(flow_0_1.shape)
#      #print(flow_0_1[0, :, 300, 150])
#      save_flow(flow_0_1, flow_0_1_file)
#      save_flow(flow_1_0, flow_1_0_file)
#      
#      flow_0_1_back = backward_warping(flow_1_0, flow_0_1)
#      flow_1_0_back = backward_warping(flow_0_1, flow_1_0)
#      occ_fw, occ_bw = occlusion(flow_0_1, flow_1_0, 1.2, 6)
#      print(occ_fw)
#      print(occ_bw)
#      a = transforms.functional.to_pil_image(occ_fw.cpu())
#      a.save('occ_fw.png')
#      b = transforms.functional.to_pil_image(occ_bw.cpu())
#      b.save('occ_bw.png')
#
#      save_flow(flow_0_1_back, flow_0_1_back_file)
#      save_flow(flow_1_0_back, flow_1_0_back_file)
#      flow_t_0 = -(1-t)*t*flow_0_1 + t*t*flow_1_0
#      flow_t_1 = (1-t)*(1-t)*flow_0_1 - t*(1-t)*flow_1_0
#
#      
#      res_0 = backward_warping(img0, flow_t_0)
#      res_1 = backward_warping(img1, flow_t_1)
#      V_t_0, V_t_1, refined_flow_t_0, refined_flow_t_1 = slomo_vision_map(img0, img1, flow_0_1, flow_1_0, flow_t_1, flow_t_0, res_0, res_1)
#      V_t_0_pic = V_t_0[0, 0, :, :]
#      V_t_1_pic = V_t_1[0, 0, :, :]
#      blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
#      V_t_0 = blur(V_t_0)
#      V_t_1 = blur(V_t_1)
#      
#      c = transforms.functional.to_pil_image(V_t_0_pic.cpu())
#      c.save('vt0.png')
#      d = transforms.functional.to_pil_image(V_t_1_pic.cpu())
#      d.save('vt1.png')
#      '''
#      _,_,H,W = img1.size()
#      H_,W_ = int(np.ceil(H/32)*32),int(np.ceil(W/32)*32)
#      pader = torch.nn.ReplicationPad2d([0, W_-W , 0, H_-H])
#      img0,img1 = pader(img0),pader(img1)
#      '''
#      print()
#      #res = backward_warping(img0, flow_1_0)
#      
#      w1 = (1-t)*occ_fw
#      w2 = t*occ_bw
#      w1 = w1.cuda()
#      w2 = w2.cuda()
#      print(w1.shape)
#      refined_res_0 = backward_warping(img0, refined_flow_t_0)
#      refined_res_1 = backward_warping(img1, refined_flow_t_1)
#       
#      #res = (w1*res_0 + w2*res_1)/(w1+w2+1e-8)
#      #res = ((1-t)*V_t_0*res_0+t*V_t_1*res_1)/((1-t)*V_t_0+t*V_t_1)
#      #V_t_0 = torch.round(V_t_0)
#      #V_t_1 = torch.round(V_t_1)
#      #V_t_0[V_t_0>0.6] = 1
#      #V_t_0[V_t_0<0.4] = 0
#      #V_t_1[V_t_1>0.6] = 1
#      #V_t_1[V_t_1<0.4] = 0
#      
#      #masked_0 = (V_t_0*refined_res_0)
#      #masked_1 = (V_t_1*refined_res_1)
#      masked_0 = (V_t_0*res_0)
#      masked_1 = (V_t_1*res_1)
#      #res = torch.zeros_like(masked_1)
#      #res = masked_1
#      res = torch.ones_like(masked_0)
#      V_size = V_t_0.size()
#      V_t_0_mask = (V_t_0 >= thre).expand(V_size[0], 3, V_size[2], V_size[3])
#      V_t_0_mask_inverse = (V_t_0 <= 1-thre).expand(V_size[0], 3, V_size[2], V_size[3])
#      
#      V_t_0_no_occ = torch.logical_and(torch.logical_not(V_t_0_mask), torch.logical_not(V_t_0_mask_inverse))
#      res[V_t_0_mask] = refined_res_0[V_t_0_mask]
#      res[V_t_0_mask_inverse] = refined_res_1[V_t_0_mask_inverse]
#      res[V_t_0_no_occ] = (0.5*refined_res_0[V_t_0_no_occ]+0.5*refined_res_1[V_t_0_no_occ])
#      #res[masked_0 == 1] = masked_0[masked_0 == 1]
#      #res[masked_1 == 1] = masked_1[masked_1 == 1]
#      #res = res/((1-t)*V_t_0 + t*V_t_1)
#      #res = masked_0 + masked_1
#      #res = (0.5*res_0+0.5*res_1)/1
#      #res = ((1-t)*V_t_0*refined_res_0+t*V_t_1*refined_res_1)/((1-t)*V_t_0+t*V_t_1)
#      #res = (0.5*V_t_0*refined_res_0+0.5*V_t_1*refined_res_1)/(0.5*V_t_0+0.5*V_t_1)
#      res = res[0,:,0:H,0:W].squeeze(0).cpu()
#
#      output = transforms.functional.to_pil_image(res)
#      #output = TP(res)
#      #output.save('./im_interp.jpg')
#      return output





if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='main function of video frame interpolation')  
	parser.add_argument('--task', default= 1., type = float, help='1 => task0, 2 => task1, 3 => task2')
	parser.add_argument('--path_to_data', default= './', help='set path to data')
	args = parser.parse_args()  
	path = args.path_to_data
	task = args.task
	path0 = './final_outdata'
	if not os.path.isdir(path0):
		os.makedirs(path0)
	if(task == 0):
		sequences = ['0']
		sequence_t = [0.5]
		sequence_name = ['frame10i11.jpg']
		pic_name1 = ['first.jpg']  
		pic_name2 = ['second.jpg']    
	elif(task == 1):  
	#task1
		sequences = ['7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
		sequence_t = [0.5]
		sequence_name = ['frame10i11.jpg']
		pic_name1 = ['frame10.jpg','frame10.jpg','frame10.jpg','frame10.jpg','frame10.jpg','frame10.jpg','frame10.jpg','frame10.jpg','frame10.jpg','frame10.jpg']  
		pic_name2 = ['frame11.jpg', 'frame11.jpg', 'frame11.jpg', 'frame11.jpg', 'frame11.jpg', 'frame11.jpg', 'frame11.jpg', 'frame11.jpg', 'frame11.jpg', 'frame11.jpg']
		path1 = './final_outdata/0_center_frame'
		if not os.path.isdir(path1):
			for seq_dir in sequences:
				os.makedirs('./final_outdata/0_center_frame/'+seq_dir+'')
		
	elif(task == 2):	
	#task2	
		sequences = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
		sequence_t = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
		sequence_name0 = ['00001.jpg', '00002.jpg', '00003.jpg', '00004.jpg', '00005.jpg', '00006.jpg', '00007.jpg']
		sequence_name1 = ['00009.jpg', '00010.jpg', '00011.jpg', '00012.jpg', '00013.jpg', '00014.jpg', '00015.jpg']
		sequence_name2 = ['00017.jpg', '00018.jpg', '00019.jpg', '00020.jpg', '00021.jpg', '00022.jpg', '00023.jpg']
		sequence_name3 = ['00025.jpg', '00026.jpg', '00027.jpg', '00028.jpg', '00029.jpg', '00030.jpg', '00031.jpg']
		sequence_name4 = ['00033.jpg', '00034.jpg', '00035.jpg', '00036.jpg', '00037.jpg', '00038.jpg', '00039.jpg']
		sequence_name5 = ['00041.jpg', '00042.jpg', '00043.jpg', '00044.jpg', '00045.jpg', '00046.jpg', '00047.jpg']
		sequence_name6 = ['00049.jpg', '00050.jpg', '00051.jpg', '00052.jpg', '00053.jpg', '00054.jpg', '00055.jpg']
		sequence_name7 = ['00057.jpg', '00058.jpg', '00059.jpg', '00060.jpg', '00061.jpg', '00062.jpg', '00063.jpg']
		sequence_name8 = ['00065.jpg', '00066.jpg', '00067.jpg', '00068.jpg', '00069.jpg', '00070.jpg', '00071.jpg']
		sequence_name9 = ['00073.jpg', '00074.jpg', '00075.jpg', '00076.jpg', '00077.jpg', '00078.jpg', '00079.jpg']
		sequence_name10 = ['00081.jpg', '00082.jpg', '00083.jpg', '00084.jpg', '00085.jpg', '00086.jpg', '00087.jpg']
		sequence_name11 = ['00089.jpg', '00090.jpg', '00091.jpg', '00092.jpg', '00093.jpg', '00094.jpg', '00095.jpg']
		#sequence_name = ['000+1.jpg', '000+2.jpg', '000+3.jpg', '000+4.jpg', '000+5.jpg', '000+6.jpg', '000+7.jpg']
		pic_name1 = ['00000.jpg', '00008.jpg', '00016.jpg', '00024.jpg', '00032.jpg', '00040.jpg', '00048.jpg', '00056.jpg', '00064.jpg', '00072.jpg', '00080.jpg', '00088.jpg']  
		pic_name2 = ['00008.jpg', '00016.jpg', '00024.jpg', '00032.jpg', '00040.jpg', '00048.jpg', '00056.jpg', '00064.jpg', '00072.jpg', '00080.jpg', '00088.jpg', '00096.jpg'] 
		path2 = './final_outdata/1_30fps_to_240fps'
		if not os.path.isdir(path2):
			for seq_dir in sequences:
				os.makedirs('./final_outdata/1_30fps_to_240fps/3/'+seq_dir+'') 
				os.makedirs('./final_outdata/1_30fps_to_240fps/4/'+seq_dir+'') 
	elif(task == 3):	
  #task3
		sequences = ['0', '1', '2', '3', '4', '5', '6', '7']
		sequence_t1 = [0.4,0.8]
		sequence_t2 = [0.2,0.6]
		sequence_name0 = ['00004.jpg', '00008.jpg']
		sequence_name1 = ['00012.jpg', '00016.jpg']
		sequence_name2 = ['00024.jpg', '00028.jpg']
		sequence_name3 = ['00032.jpg', '00036.jpg']
		sequence_name4 = ['00044.jpg', '00048.jpg']
		sequence_name5 = ['00052.jpg', '00056.jpg']
		sequence_name6 = ['00064.jpg', '00068.jpg']
		sequence_name7 = ['00072.jpg', '00076.jpg']
		pic_name1 = ['00000.jpg', '00010.jpg', '00020.jpg', '00030.jpg', '00040.jpg', '00050.jpg', '00060.jpg', '00070.jpg']  
		pic_name2 = ['00010.jpg', '00020.jpg', '00030.jpg', '00040.jpg', '00050.jpg', '00060.jpg', '00070.jpg', '00080.jpg'] 
		path3 = './final_outdata/2_24fps_to_60fps'
		if not os.path.isdir(path3):
			for seq_dir in sequences:
				os.makedirs('./final_outdata/2_24fps_to_60fps/3/'+seq_dir+'') 
				os.makedirs('./final_outdata/2_24fps_to_60fps/4/'+seq_dir+'') 
	elif(task == 4):	
	#task2	
		sequences = ['0']
		sequence_t = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
		sequence_name0 = ['00001.jpg', '00002.jpg', '00003.jpg', '00004.jpg', '00005.jpg', '00006.jpg', '00007.jpg']
		sequence_name1 = ['00009.jpg', '00010.jpg', '00011.jpg', '00012.jpg', '00013.jpg', '00014.jpg', '00015.jpg']
		sequence_name2 = ['00017.jpg', '00018.jpg', '00019.jpg', '00020.jpg', '00021.jpg', '00022.jpg', '00023.jpg']
		sequence_name3 = ['00025.jpg', '00026.jpg', '00027.jpg', '00028.jpg', '00029.jpg', '00030.jpg', '00031.jpg']
		sequence_name4 = ['00033.jpg', '00034.jpg', '00035.jpg', '00036.jpg', '00037.jpg', '00038.jpg', '00039.jpg']
		sequence_name5 = ['00041.jpg', '00042.jpg', '00043.jpg', '00044.jpg', '00045.jpg', '00046.jpg', '00047.jpg']
		sequence_name6 = ['00049.jpg', '00050.jpg', '00051.jpg', '00052.jpg', '00053.jpg', '00054.jpg', '00055.jpg']
		sequence_name7 = ['00057.jpg', '00058.jpg', '00059.jpg', '00060.jpg', '00061.jpg', '00062.jpg', '00063.jpg']
		sequence_name8 = ['00065.jpg', '00066.jpg', '00067.jpg', '00068.jpg', '00069.jpg', '00070.jpg', '00071.jpg']
		sequence_name9 = ['00073.jpg', '00074.jpg', '00075.jpg', '00076.jpg', '00077.jpg', '00078.jpg', '00079.jpg']
		sequence_name10 = ['00081.jpg', '00082.jpg', '00083.jpg', '00084.jpg', '00085.jpg', '00086.jpg', '00087.jpg']
		sequence_name11 = ['00089.jpg', '00090.jpg', '00091.jpg', '00092.jpg', '00093.jpg', '00094.jpg', '00095.jpg']
		pic_name1 = ['00000.jpg', '00008.jpg', '00016.jpg', '00024.jpg', '00032.jpg', '00040.jpg', '00048.jpg', '00056.jpg', '00064.jpg', '00072.jpg', '00080.jpg', '00088.jpg']  
		pic_name2 = ['00008.jpg', '00016.jpg', '00024.jpg', '00032.jpg', '00040.jpg', '00048.jpg', '00056.jpg', '00064.jpg', '00072.jpg', '00080.jpg', '00088.jpg', '00096.jpg']   
	if(task == 2 or task == 3):
		directory = ['3','4']
	else:
		directory = ['3']
	for dire in directory:  
		
		for sq, pic1, pic2 in zip(sequences, pic_name1, pic_name2):
			if(task == 0):
				arguments_strFirst = './images/first.png'
				arguments_strSecond = './images/second.png'        
			elif(task == 1):
				arguments_strFirst = './'+path+'/0_center_frame/'+sq+'/input/frame10.png'
				arguments_strSecond = './'+path+'/0_center_frame/'+sq+'/input/frame11.png'
			elif(task == 2):
				arguments_strFirst = './'+path+'/1_30fps_to_240fps/'+dire+'/'+sq+'/input/'+pic1+''
				arguments_strSecond = './'+path+'/1_30fps_to_240fps/'+dire+'/'+sq+'/input/'+pic2+''
			elif(task == 3):
				arguments_strFirst = './'+path+'/2_24fps_to_60fps/'+dire+'/'+sq+'/input/'+pic1+''
				arguments_strSecond = './'+path+'/2_24fps_to_60fps/'+dire+'/'+sq+'/input/'+pic2+''
			elif(task == 4):
				arguments_strFirst = './100/'+sq+'/input/'+pic1+''
				arguments_strSecond = './100/'+sq+'/input/'+pic2+''
			tenFirst = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
			tenSecond = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
	
			_,H,W = tenSecond.size()
			input_ts = torch.zeros((1,3,H,W)).cuda()
			input_ts[0,:,:,:] = tenSecond[:,:,:]
			gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
			gridX = torch.tensor(gridX, requires_grad=False).cuda()
			gridY = torch.tensor(gridY, requires_grad=False).cuda()
			tenOutput = estimate(tenFirst, tenSecond)
			u = tenOutput[:,0,:,:]
			v = tenOutput[:,1,:,:]
			x = gridX.unsqueeze(0).expand_as(u).float()+u.cuda()
			y = gridY.unsqueeze(0).expand_as(v).float()+v.cuda()
			normx = 2*(x/W-0.5)
			normy = 2*(y/H-0.5)
			grid = torch.stack((normx,normy), dim=3)
			#tenOutputo_nn = torch.nn.functional.grid_sample(input_ts, grid)
			#tenOutputo = tenOutputo_nn[0,:,0:H,0:W].squeeze(0).cpu() 
			#tenOutputo = tenOutputo[[2,1,0],:,:]
			#tenOutputo = transforms.functional.to_pil_image(tenOutputo)
			#tenOutputo.save('./im_sectofir.png')
		
		
			_,H1,W1 = tenFirst.size()
			input_ts1 = torch.zeros((1,3,H1,W1)).cuda()
			input_ts1[0,:,:,:] = tenFirst[:,:,:]
			gridX1, gridY1 = np.meshgrid(np.arange(W1), np.arange(H1))
			gridX1 = torch.tensor(gridX1, requires_grad=False).cuda()
			gridY1 = torch.tensor(gridY1, requires_grad=False).cuda()
			tenOutput1 = estimate(tenSecond, tenFirst)
			u1 = tenOutput1[:,0,:,:]
			v1 = tenOutput1[:,1,:,:]
			x1 = gridX1.unsqueeze(0).expand_as(u1).float()+u1.cuda()
			y1 = gridY1.unsqueeze(0).expand_as(v1).float()+v1.cuda()
			normx1 = 2*(x1/W1-0.5)
			normy1 = 2*(y1/H1-0.5)
			grid1 = torch.stack((normx1,normy1), dim=3)
			#tenOutputo1_nn = torch.nn.functional.grid_sample(input_ts1, grid1)  
			#tenOutputo1 = tenOutputo1_nn[0,:,0:H1,0:W1].squeeze(0).cpu() 
			#tenOutputo1 = tenOutputo1[[2,1,0],:,:]
			#tenOutputo1 = transforms.functional.to_pil_image(tenOutputo1)
			#tenOutputo1.save('./im_firtosec.png')
		
		#backward slomo
			t = torch.tensor(0.6) 
			Flow_t_0 = -(1-t)*t*tenOutput+t*t*tenOutput1
			Flow_t_1 = (1-t)*(1-t)*tenOutput-t*(1-t)*tenOutput1
			u3 = Flow_t_1[:,0,:,:]
			v3 = Flow_t_1[:,1,:,:]
			x3 = gridX.unsqueeze(0).expand_as(u3).float()+u3.cuda()
			y3 = gridY.unsqueeze(0).expand_as(v3).float()+v3.cuda()
			normx3 = 2*(x3/W-0.5)
			normy3 = 2*(y3/H-0.5)
			grid3 = torch.stack((normx3,normy3), dim=3)
			#slomo_ans0 = torch.nn.functional.grid_sample(input_ts, grid3)
			#slomo_ans0 = slomo_ans0[0,:,0:H,0:W].squeeze(0).cpu() 
			#slomo_ans0 = slomo_ans0[[2,1,0],:,:]
			#slomo_ans0 = transforms.functional.to_pil_image(slomo_ans0)
			#slomo_ans0.save('./im_slomo_firtosec.jpg')
		
			u4 = Flow_t_0[:,0,:,:]
			v4 = Flow_t_0[:,1,:,:]
			x4 = gridX1.unsqueeze(0).expand_as(u4).float()+u4.cuda()
			y4 = gridY1.unsqueeze(0).expand_as(v4).float()+v4.cuda()
			normx4 = 2*(x4/W1-0.5)
			normy4 = 2*(y4/H1-0.5)
			grid4 = torch.stack((normx4,normy4), dim=3)
			slomo_ans1 = torch.nn.functional.grid_sample(input_ts1, grid4)
			#slomo_ans1 = slomo_ans1[0,:,0:H1,0:W1].squeeze(0).cpu() 
			#slomo_ans1 = slomo_ans1[[2,1,0],:,:]
			#slomo_ans2 = slomo_ans1[[2,1,0],:,:]
			#slomo_ans1 = transforms.functional.to_pil_image(slomo_ans1)
			#slomo_ans1.save('./im_slomo_sectofir.jpg')
	
		#forward 
			#input_for = torch.zeros((1,H,W,2)).cuda()
			#input_for_1 = torch.ones((1,H,W,2)).cuda()
			#grid4_long = grid4.long().cuda()
			#print(grid4_long.size())
			#print(input_ts1.size())
		#for part 1
			#tenOutput_compare01 = torch.zeros((1,2,H,W)).cuda()
			#tenOutput_compare10 = torch.zeros((1,2,H,W)).cuda()
			#tenOutput_process01 = torch.zeros((1,2,H,W)).cuda()     
			#tenOutput_process01[:,:,:,:] = tenOutput[:,:,:,:]    
			#tenOutput_process10 = torch.zeros((1,2,H,W)).cuda()     
			#tenOutput_process10[:,:,:,:] = tenOutput1[:,:,:,:]  
			#tenOutput_1_0_back = torch.nn.functional.grid_sample(tenOutput1,grid1)
			#tenOutput_0_1_back = torch.nn.functional.grid_sample(tenOutput_process10,grid)
			occl0 = np.zeros((H,W))
			occl1 = np.zeros((H,W))
			#x = torch.tensor(-1).cpu()
			threshold0 = 4 #9.5
			threshold1 = 3 #6.4
			par1 = 5 #9.5
			par2 = 15 #9.5            
			count = 1
			for a in range(H):
				for b in range(W):
					y = np.int(np.round(tenOutput1[0,1,a,b])) + a
					x = np.int(np.round(tenOutput1[0,0,a,b])) + b 
					if(x>W-1 or y>H-1):
						occl1[a,b] = -1  
					else:      
						if(tenOutput1[0,0,a,b] + tenOutput[0,0,y,x].cpu() < threshold0 and tenOutput1[0,1,a,b] + tenOutput[0,1,y,x].cpu() < threshold0):
							occl1[a,b] = 1
							count = count + 1		
						else:				
							occl1[a,b] = -1
			#		if(x>W-1 or y>H-1):
			#			occl1[a,b] = -1 + 2*occl0[a,b]  
			#		else:      
			#			if(tenOutput1[0,0,a,b] + tenOutput[0,0,y,x].cpu() < threshold1 and tenOutput1[0,1,a,b] + tenOutput[0,1,y,x].cpu() < threshold1):
			#				occl1[a,b] = 1 + 2*occl0[a,b]
			#				count = count + 1		
			#			else:				
			#				occl1[a,b] = -1 + 2*occl0[a,b]
			#print(count) 
			#occl1[:,:] = 2*occl1[:,:] + occl0[:,:] 
			#occl1[:,:] = xip.jointBilateralFilter(np.array(occl1).astype(np.float32),np.array(occl1).astype(np.float32),d = 14,sigmaColor = 62,sigmaSpace = 8)	
			#occl0 = cv2.GaussianBlur(np.array(occl0.cpu()),(5,5),3)
			occl1 = cv2.GaussianBlur(np.array(occl1),(7,7),0) 
			#occl1 = cv2.GaussianBlur(np.array(occl1),(5,5),0) 
			#occl1 = cv2.GaussianBlur(np.array(occl1),(3,3),0)  
			#occl1 = cv2.medianBlur(occl1,5)                     
			if(task == 2 or task == 4):				
				if(sq == '0'):
					sequence_name = sequence_name0
				elif(sq == '1'):
					sequence_name = sequence_name1				
				elif(sq == '2'):
					sequence_name = sequence_name2				
				elif(sq == '3'):
					sequence_name = sequence_name3
				elif(sq == '4'):
					sequence_name = sequence_name4
				elif(sq == '5'):
					sequence_name = sequence_name5
				elif(sq == '6'):
					sequence_name = sequence_name6
				elif(sq == '7'):
					sequence_name = sequence_name7				
				elif(sq == '8'):
					sequence_name = sequence_name8				
				elif(sq == '9'):
					sequence_name = sequence_name9
				elif(sq == '10'):
					sequence_name = sequence_name10
				else:
					sequence_name = sequence_name11


			if(task == 3):
				if(sq == '0' or sq == '2' or sq == '4' or sq == '6'):
					sequence_t = sequence_t1
				else:
					sequence_t = sequence_t2  
				
				if(sq == '0'):
					sequence_name = sequence_name0
				elif(sq == '1'):
					sequence_name = sequence_name1				
				elif(sq == '2'):
					sequence_name = sequence_name2				
				elif(sq == '3'):
					sequence_name = sequence_name3
				elif(sq == '4'):
					sequence_name = sequence_name4
				elif(sq == '5'):
					sequence_name = sequence_name5
				elif(sq == '6'):
					sequence_name = sequence_name6
				else:
					sequence_name = sequence_name7
			#interp_frame(tenFirst, tenSecond)          
			#tenSoftmax_o[:,:,:] = torch.zeros((3,H,W)).cuda()
			#tenSoftmax_o[:,:,:] = input_ts[0,[2,1,0],:,:]
			tenfirst = torch.FloatTensor(np.ascontiguousarray(cv2.imread(filename=arguments_strFirst, flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()
			tensecond = torch.FloatTensor(np.ascontiguousarray(cv2.imread(filename=arguments_strSecond, flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()		
			for timet, name in zip(sequence_t,sequence_name):
				tenMetric = torch.nn.functional.l1_loss(input=tenfirst, target=backwarp(tenInput=tensecond, tenFlow=tenOutput.cuda()), reduction='none').mean(1, True)
				tenSoftmax = softsplat.FunctionSoftsplat(tenInput=tenfirst, tenFlow=tenOutput.cuda() * timet, tenMetric=-20 *tenMetric, strType='softmax') # -20.0 is a hyperparameter, called 'alpha' in the paper, that could be learned using a torch.Parameter
				tenSoftmax = tenSoftmax[0,:,0:H,0:W].squeeze(0).cpu() 
				tenSoftmax = tenSoftmax[[2,1,0],:,:]
				#for a in range(H):
				#	for b in range(W):
				#		if(occl0[a,b] < 0 and occl1[a,b] > 0):
				#			tenSoftmax[:,a,b] = slomo_ans2[[2,1,0],a,b]
				#		elif(occl0[a,b] > 0 and occl1[a,b] < 0):
				#			tenSoftmax[:,a,b] = slomo_ans2[[2,1,0],a,b]  
				#		elif(occl0[a,b] < 0 and occl1[a,b] < 0):
				#			tenSoftmax[:,a,b] = slomo_ans2[[2,1,0],a,b]
				for a in range(H):
					for b in range(W):
						if(occl1[a,b] == -1):
							tenSoftmax[:,a,b] = input_ts[0,[2,1,0],a,b]
        #      #tenSoftmax[:,a,b] = (input_ts[0,[2,1,0],a,b] + slomo_ans2[[2,1,0],a,b])/2
				#		#elif(occl1[a,b] < 0 and occl1[a,b] > -0.3):
				#		#	tenSoftmax[:,a,b] = input_ts1[0,[2,1,0],a,b]
				#		#lif(occl1[a,b] == -1 and occl1[a,b] == -1):
				#		#	tenSoftmax[:,a,b] = (input_ts[0,[2,1,0],a,b]+input_ts1[0,[2,1,0],a,b])/2

				#tenSoftmax[:,:,:] = xip.jointBilateralFilter(np.array(slomo_ans2).astype(np.float32),np.array(tenSoftmax).astype(np.float32),d = 14,sigmaColor = 62,sigmaSpace = 8)
						#	if(occl0[a,b]<occl1[a,b]):              
						#		tenSoftmax[:,a,b] = (input_ts[0,[2,1,0],a,b]*9 + input_ts1[0,[2,1,0],a,b]*1)/10
						#	else:              
						#		tenSoftmax[:,a,b] = (input_ts[0,[2,1,0],a,b]*1 + input_ts1[0,[2,1,0],a,b]*9)/10 
						#	#tenSoftmax[:,a,b] = input_ts[0,[2,1,0],a,b]                             
				tenSoftmax = transforms.functional.to_pil_image(tenSoftmax)
				if(task == 0): 
					tenSoftmax.save('./frame10i11.jpg')                 
				elif(task == 1):
					tenSoftmax.save('./final_outdata/0_center_frame/'+sq+'/'+name+'')
				elif(task == 2):
					tenSoftmax.save('./final_outdata/1_30fps_to_240fps/'+dire+'/'+sq+'/'+name+'')
				elif(task == 3):
					tenSoftmax.save('./final_outdata/2_24fps_to_60fps/'+dire+'/'+sq+'/'+name+'')
				elif(task == 4):
					tenSoftmax.save('./task1_out/'+name+'')
	#frame0 = cv2.imread('./images/first.png')
	#frame1 = cv2.imread('./images/second.png')
	#i1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
	#i2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	#flow0 = cv2.calcOpticalFlowFarneback(i1, i2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	#frame_t= internp(frame0=frame0, frame1=frame1, t=0.5, flow0=flow0)
	#slomo_ans1 = frame_t[0,:,0:H1,0:W1].squeeze(0).cpu() 
	#slomo_ans1 = slomo_ans1[[2,1,0],:,:]
	#frame_t = transforms.functional.to_pil_image(frame_t)
	#frame_t.save('./forward.jpg')
	#cv2.imwrite(filename='./forward.jpg', img=(frame_t * 1.0).clip(0.0, 255.0).astype(np.uint8))

	#numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
	#numpy.array([ tenOutput1.shape[2], tenOutput1.shape[1] ], numpy.int32).tofile(objOutput)
	#numpy.array(tenOutput1.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)
	
	#objOutput.close()
# end