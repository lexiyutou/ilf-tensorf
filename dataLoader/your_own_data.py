import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T


from .ray_utils import *

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m   #(3,4)
# nerf_pytorch: load_llff poses_avg and recenter_poses

def poses_avg(poses):  #(N,3,4)or (N,4,4)
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)

    return c2w

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da=da/np.linalg.norm(da)
	db=db/np.linalg.norm(db)
	c=np.cross(da,db)
	denom=(np.linalg.norm(c)**2)
	t=ob-oa
	ta=np.linalg.det([t,db,c])/(denom+1e-10)
	tb=np.linalg.det([t,da,c])/(denom+1e-10)
	if ta>0:
		ta=0
	if tb>0:
		tb=0
	return (oa+ta*da+ob+tb*db)*0.5,denom

def recenter_poses(poses):  #(N,4,4)

    poses_ = poses+0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    return poses

class YourOwnDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1,recenter=True,spheretrans=True):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        
        
        #newly added
        self.recenter = recenter #recenter poses
        self.spheretrans = spheretrans #translate the poses to move the ball center to origion
        self.pose_avg = None  #average pose
        self.scale = 1.0
        self.ball_center = np.array([0.0,0.0,-3.6])
        self.ov_pts = None
        
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-15.0, -15.0, -5.0], [15.0, 15.0, 25.0]])
        #scene_bbox for ruv
        # self.scene_bbox = torch.tensor([[-30.0,-30.0,-30.0],[30.0,30.0,30.0]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        # self.define_proj_mat()

        self.white_bg = True
        self.near_far = [0.0,30.0]
        # self.near_far = [1.0,30.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth


    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        self.image_paths = []
        self.poses = []

        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []

        poses_np = []

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        
        
        # load extrinsics 
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):
            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]
            poses_np.append(pose)
        
        #recenter poses
        if self.recenter:
            poses_np = np.array(poses_np)
            poses_np = recenter_poses(poses_np)  #[N,4,4]
            self.pose_avg = poses_avg(poses_np)
            print("pose_avg",self.pose_avg)
                   
        #translate poses
        if self.spheretrans:
            # find a central point they are all looking at
            print("computing center of the dome...")
            totw=0
            totp=[0,0,0]
            for f in poses_np:
                mf=f[0:3,:]
                for g in poses_np:
                    mg=g[0:3,:]
                    p,w=closest_point_2_lines(mf[:,3],mf[:,2],mg[:,3],mg[:,2])
                    if w>0.01:
                        totp+=p*w
                        totw+=w
            totp/=totw
            print(totp) # the cameras are looking at totp
            
            self.ball_center = np.array(totp)

            poses_np[:,:3,-1] -=self.ball_center
            # visualize_poses(poses_np)   
                 
        #scale the poses to make sure it's in the [-1,1]^3 cube
        self.scale = 1./np.max(np.abs(poses_np[:,:3,3]))
        poses_np[:,:3,3] *= self.scale
        print("poses",poses_np[:,:,-1])
        
        self.poses = torch.FloatTensor(poses_np)
                    
        #load image and meta data one by one
        # not_train_list = [0,1,2,3]
        not_train_list =[]
        # pano = np.zeros((h,w,3))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):
            #img_list:#
            if i in not_train_list and self.split=="train":
                print("for training: get away with the first image")
                continue
            # print("check i:",i)
            frame = self.meta['frames'][i]
            image_path = os.path.join(self.root_dir, f"{frame['file_path']}")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            
            w, h = int(frame['w']/self.downsample), int(frame['h']/self.downsample)
            self.img_wh = [w,h]
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(-1, w*h).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]
            
            cx = frame['cx']/self.downsample
            cy = frame['cy']/self.downsample
            focal_x = frame['fl_x']/self.downsample
            focal_y = frame['fl_y']/self.downsample
            directions = get_ray_directions(h, w, [focal_x,focal_y], center=[cx, cy])  # (h, w, 3)
            directions = directions/torch.norm(directions,dim=-1,keepdim = True)
            intrinsics = torch.tensor([[focal_x,0,cx],[0,focal_y,cy],[0,0,1]]).float()
            rays_o, rays_d = get_rays(directions, self.poses[i])  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)
        
        if self.split =='test':    
            directions = get_ray_directions(h, w, [focal_x,focal_y], center=[cx, cy])
            self.directions = directions/torch.norm(directions,dim=-1,keepdim = True)
            new_poses = self.poses
            new_poses[:,2,-1] -=0.6
            self.render_path = new_poses
            print("******* render path shape",self.render_path.shape)
            
        
 
            


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
    '''
    def computeAffine(self,cam_intrin,cam_c2w,ref_intrin,ref_c2w):
        return np.array(ref_intrin @ torch.inverse(ref_c2w)[:3,:] @ cam_c2w[:,:3] @ torch.inverse(cam_intrin))
    
    def computesinglepano(self,img,cam_intrin,cam_c2w,ref_intrin,ref_c2w):
        AffineMatrix = self.computeAffine(cam_intrin,cam_c2w,ref_intrin,ref_c2w)
        pano_W = 5760
        pano_H = 3240
        pano = cv2.warpAffine(img,AffineMatrix[:2,:],(pano_W,pano_H))
        return pano
    '''
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img}
        return sample
