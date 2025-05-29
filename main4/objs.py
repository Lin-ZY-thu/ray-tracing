import torch
from func import dot, norm, reflect,raytrace,cross
from config import device
from setting import depth,ambient,light_sources

class Sphere:
    def __init__(self, center, radius, diffuse,texture=None, mirror=0.0):
        self.center = torch.tensor(center, device=device)
        self.radius = radius
        self.diffuse = torch.tensor(diffuse, device=device)
        self.mirror = mirror
        self.texture = texture

    def intersect(self, O, D):
        oc = O - self.center
        a = dot(D, D)
        b = 2 * dot(oc, D)
        c = dot(oc, oc) - self.radius**2
        disc = b**2 - 4 * a * c
        sqrt_disc = torch.sqrt(torch.clamp_min(disc, 0))
        root1 = (-b - sqrt_disc) / (2 * a)
        root2 = (-b + sqrt_disc) / (2 * a)
        valid = (disc > 0) & (root1 > 1e-4)
        return torch.where(valid, root1, torch.tensor(torch.inf, device=device))

    def shade(self, O, D, t, scene, bounce=0):
        M = O + D * t
        N = norm(M - self.center)
        toL = norm(O - M)
        
        if self.texture:
            texture_color = self.texture(M, self.center, self.diffuse)
        else:
            texture_color = self.diffuse
        
        #
        diffuse_contribution =0.007* torch.clamp_min(dot(N, toL), 0)
        for light in light_sources:
            # 计算法线与光源方向的点积
            light_dir = norm(light['position'] - M)
            # 计算漫反射贡献
            diffuse_contribution += light['intensity'] * torch.clamp_min(dot(N, light_dir), 0)
        diffuse = texture_color * diffuse_contribution
        #

        # 漫反射
        #diffuse = texture_color * torch.clamp_min(dot(N, toL), 0)
        
        # 镜面反射
        reflection = torch.zeros_like(diffuse)
        if bounce < depth and self.mirror > 0:
            R = reflect(D, N)
            reflection = raytrace(M + N*1e-4, R, scene, bounce+1) * self.mirror
        
        # 环境光
        ambient1 = self.diffuse * ambient
        
        return ambient1 + diffuse + reflection

class Plane:
    def __init__(self, center, normal, diffuse,texture=None, mirror=0.0):
        self.center = torch.tensor(center, device=device)
        self.normal = torch.tensor(normal, device=device)
        self.diffuse = torch.tensor(diffuse, device=device)
        self.mirror = mirror
        self.texture = texture 

    def intersect(self, O, D):
        dn = dot(D, self.normal)
        d = dot(self.center - O, self.normal) / dn
        pred = (d > 0)&(abs(dn)>0.01)
        return torch.where(pred, d, torch.tensor(torch.inf, device=device))

    def shade(self, O, D, t, scene, bounce=0):
        M = O + D * t
        N = self.normal
        toL = norm(O - M)
        
        # 棋盘纹理
        #diffuse = self.diffuse * torch.clamp_min(dot(N, toL), 0) * (0.9 * dot((M - self.center), (M - self.center)) * 10 % 2 + 0.1)
        if self.texture:
            texture_color = self.texture(M, self.center, self.diffuse)
        else:
            texture_color = self.diffuse
        
        #
        diffuse_contribution =0.007* torch.clamp_min(dot(N, toL), 0)
        for light in light_sources:
            # 计算法线与光源方向的点积
            light_dir = norm(light['position'] - M)
            # 计算漫反射贡献
            diffuse_contribution += light['intensity'] * torch.clamp_min(dot(N, light_dir), 0)
        diffuse = texture_color * diffuse_contribution
        #
        # 漫反射
        #diffuse = texture_color * torch.clamp_min(dot(N, toL), 0) 
        
        # 反射
        reflection = torch.zeros_like(diffuse)
        if bounce < depth and self.mirror > 0:
            R = reflect(D, N)
            reflection = raytrace(M + N * 1e-4, R, scene, bounce + 1) * self.mirror
        
        ambient1=self.diffuse*ambient

        return diffuse + reflection + ambient1


class Cylinder:
    def __init__(self, center, normal, radius, height, diffuse, texture=None, mirror=0.0):
        self.center = torch.tensor(center, device=device)
        self.normal = torch.tensor(normal, device=device)
        self.radius = radius
        self.height = height
        self.diffuse = torch.tensor(diffuse, device=device)
        self.texture = texture
        self.mirror = mirror

    def intersect(self, O, D):
        # 将光线坐标转换到圆柱体的局部坐标系
        w = norm(self.normal)
        v = norm(cross(D, w))
        u = norm(cross(w, v))
        local_O = O - self.center
        local_D = D

        # 计算光线在局部坐标系中的参数
        ox = dot(local_O, u)
        oy = dot(local_O, v)
        oz = dot(local_O, w)
        dx = dot(local_D, u)
        dy = dot(local_D, v)
        dz = dot(local_D, w)

        # 计算与圆柱体侧面的交点
        a = dx**2 + dy**2
        b = 2 * (dx * ox + dy * oy)
        c = ox**2 + oy**2 - self.radius**2
        disc = b**2 - 4 * a * c

        # 初始化 side_hit 为无穷大 
        side_hit = torch.full_like(ox, torch.inf)

        # 检查判别式是否为非负值  如果没有会出现条带
        valid = disc >= 0
        if valid.any():
            sqrt_disc = torch.sqrt(disc[valid])
            root1 = (-b[valid] - sqrt_disc) / (2 * a[valid])
            root2 = (-b[valid] + sqrt_disc) / (2 * a[valid])

            # 检查交点是否在圆柱体的高度范围内
            z1 = oz[valid] + dz[valid] * root1
            z2 = oz[valid] + dz[valid] * root2
            valid_root1 = (root1 > 1e-4) & (z1 >= 0) & (z1 <= self.height)
            valid_root2 = (root2 > 1e-4) & (z2 >= 0) & (z2 <= self.height)

            # 计算有效交点
            hit1 = torch.where(valid_root1, root1, torch.tensor(torch.inf, device=device))
            hit2 = torch.where(valid_root2, root2, torch.tensor(torch.inf, device=device))
            side_hit_valid = torch.minimum(hit1, hit2)
            side_hit[valid] = side_hit_valid

        # 计算与圆柱体顶面和底面的交点
        t_top = (self.height - oz) / dz
        t_bottom = -oz / dz
        top_hit = torch.where((t_top > 1e-4) & ((ox + dx * t_top)**2 + (oy + dy * t_top)**2 <= self.radius**2), t_top, torch.tensor(torch.inf, device=device))
        bottom_hit = torch.where((t_bottom > 1e-4) & ((ox + dx * t_bottom)**2 + (oy + dy * t_bottom)**2 <= self.radius**2), t_bottom, torch.tensor(torch.inf, device=device))

        # 返回最近的交点
        return side_hit#torch.minimum(torch.minimum(side_hit, top_hit), bottom_hit)

    def shade(self, O, D, t, scene, bounce=0):
        M = O + D * t
        N = norm((M - self.center)-dot((M - self.center),self.normal)*self.normal)
        toL = norm(O - M)
        
        if self.texture:
            texture_color = self.texture(M, self.center, self.diffuse)
        else:
            texture_color = self.diffuse
        
        #
        diffuse_contribution =0.007* torch.clamp_min(dot(N, toL), 0)
        for light in light_sources:
            # 计算法线与光源方向的点积
            light_dir = norm(light['position'] - M)
            # 计算漫反射贡献
            diffuse_contribution += light['intensity'] * torch.clamp_min(dot(N, light_dir), 0)
        diffuse = texture_color * diffuse_contribution
        #
        # 漫反射
        #diffuse = texture_color * torch.clamp_min(dot(N, toL), 0) 
        
        # 镜面反射
        reflection = torch.zeros_like(diffuse)
        if bounce < depth and self.mirror > 0:
            R = reflect(D, N)
            reflection = raytrace(M + N*1e-4, R, scene, bounce+1) * self.mirror
        
        # 环境光
        ambient1 = self.diffuse * ambient
        
        return ambient1 + diffuse + reflection
