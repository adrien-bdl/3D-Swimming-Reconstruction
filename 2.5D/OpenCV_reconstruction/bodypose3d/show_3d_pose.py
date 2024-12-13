import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import DLT, ensure_directory_exists, write_videos_to_disk

plt.style.use('seaborn')

pose_keypoints = np.array([16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28])


def read_keypoints(filename):
	fin = open(filename, 'r')

	kpts = []
	while (True):
		line = fin.readline()
		if line == '': break

		line = line.split()
		line = [float(s) for s in line]

		line = np.reshape(line, (len(pose_keypoints), -1))
		kpts.append(line)

	kpts = np.array(kpts)
	return kpts


def visualize_3d(p3ds, window_size=200, slow=False):
	"""Now visualize in 3D"""
	torso = [[0, 1], [1, 7], [7, 6], [6, 0]]
	armr = [[1, 3], [3, 5]]
	arml = [[0, 2], [2, 4]]
	legr = [[6, 8], [8, 10]]
	legl = [[7, 9], [9, 11]]
	body = [torso, arml, armr, legr, legl]
	colors = ['red', 'blue', 'green', 'black', 'orange']

	from mpl_toolkits.mplot3d import Axes3D

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	# ax.elev = 180  #95
	# ax.azim = 0  #90
	# ax.roll = -90
	ax.elev = -50
	ax.azim = 0
	ax.roll = -90

	for framenum, kpts3d in enumerate(p3ds):
		# if framenum % 2 == 0: continue  #skip every 2nd frame
		for bodypart, part_color in zip(body, colors):
			for _c in bodypart:
				ax.plot(xs=[kpts3d[_c[0], 0], kpts3d[_c[1], 0]],
				        ys=[kpts3d[_c[0], 1], kpts3d[_c[1], 1]],
				        zs=[kpts3d[_c[0], 2], kpts3d[_c[1], 2]],
				        linewidth=4,
				        c=part_color)

		# ax.elev += rate
		# ax.azim += rate
		#uncomment these if you want scatter plot of keypoints and their indices.
		# for i in range(12):
		#     #ax.text(kpts3d[i,0], kpts3d[i,1], kpts3d[i,2], str(i))
		#     #ax.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])

		#ax.set_axis_off()

		# ax.set_xticks([])
		# ax.set_yticks([])
		# ax.set_zticks([])

		ax.set_xlim3d(-window_size, window_size)
		ax.set_xlabel('x')
		ax.set_ylim3d(-window_size, window_size)
		ax.set_ylabel('y')
		ax.set_zlim3d(-window_size, window_size)
		ax.set_zlabel('z')

		plt.savefig(f"out/3d_pose/frame_{framenum}.jpg", format="jpg")

		plt.pause(1 if slow else .1)
		ax.cla()
	print(f"{ax.azim=}", f"{ax.elev=}")
	plt.close(fig)


if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))

	ensure_directory_exists('out/3d_pose')

	p3ds = read_keypoints('out/kpts_3d.dat')
	visualize_3d(p3ds)

	write_videos_to_disk("out/3d_pose/frame_%d.jpg", "out/3d_pose.mp4", fps=7)
