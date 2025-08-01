import cv2
import numpy as np
import math
import time
from RRT import RRT
import torch
import torchvision.transforms as transforms
show_animation = False

class RRTStar(RRT):
    """
    Class for RRT Star planning
    """
    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=30.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=300,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False,
                 robot_radius=1.0):
        super().__init__(start, goal, obstacle_list, rand_area, expand_dis,
                         path_resolution, goal_sample_rate, max_iter,
                         robot_radius=robot_radius)
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.search_until_max_iter = search_until_max_iter
        self.node_list = []
        self.obstacles = []
        default_size = 1  # Replace with an appropriate size value if necessary

        for x in range(len(self.obstacle_list)):  # cost_map should be your 2D grid of zeros and ones
            for y in range(len(self.obstacle_list[0])):
                if self.obstacle_list[x][y] == 1:  # Assuming 1 indicates an obstacle
                    self.obstacles.append((x, y, default_size))

    def planning(self, animation=True):
        self.node_list = [self.start]

        

        for i in range(self.max_iter):
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd,
                                  self.expand_dis)
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + \
                math.hypot(new_node.x - near_node.x,
                           new_node.y - near_node.y)

            if self.check_collision(new_node, self.obstacles, self.robot_radius):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if animation:
                self.draw_graph(rnd)

            if (not self.search_until_max_iter) and new_node:
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)
        return None

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacles, self.robot_radius):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))
        min_cost = min(costs)
        if min_cost == float("inf"):
            return None
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost
        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis]
        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(t_node, self.obstacles, self.robot_radius):
                safe_goal_inds.append(goal_ind)
        if not safe_goal_inds:
            return None
        safe_goal_costs = [self.node_list[i].cost + self.calc_dist_to_goal(self.node_list[i].x, self.node_list[i].y)
                           for i in safe_goal_inds]
        min_cost = min(safe_goal_costs)
        for i, cost in zip(safe_goal_inds, safe_goal_costs):
            if cost == min_cost:
                return i
        return None

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)
            no_collision = self.check_collision(edge_node, self.obstacles, self.robot_radius)
            improved_cost = near_node.cost > edge_node.cost
            if no_collision and improved_cost:
                for node in self.node_list:
                    if node.parent == self.node_list[i]:
                        node.parent = edge_node
                self.node_list[i] = edge_node
                self.propagate_cost_to_leaves(self.node_list[i])

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)
