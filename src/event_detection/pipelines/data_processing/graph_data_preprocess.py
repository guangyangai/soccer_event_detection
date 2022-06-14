"""
Author: Guang Yang
Write out shot data to load into graph
"""
import sys
import os
import re
import time
import json
import random
import numpy as np
import pandas as pd
import networkx as nx
import gc

sys.path.insert(0, '/home/ec2-user/SageMaker/NHLShotQuality/src/')
sys.path.insert(0, '/home/ec2-user/SageMaker/NHLShotQuality/src/features/')
from utils import print_dict
from utils import save_json_file
from utils import load_json_file
from utils import save_to_pickle
from utils import load_pickle
from utils import ensure_directory
from utils import path_join


NODE_COLOR_DICT = {
    'puck':'#1f77b4','shooter':'#ff7f0e','closest_defender':'#9467bd','goalie':'#d62728','shot_assister':'#2ca02c','non_goalie_shot_blocker':'#8c564b','other_offend_player':'#7f7f7f','other_defend_player':'#e377c2',
                  }
NODE_INDEX_DICT = {
    'puck':0,'shooter':1,'closest_defender':2,'goalie':3,'shot_assister':4,'non_goalie_shot_blocker':5,
                  }
NUM_ENTITIES = 13
outcome_label = {'ShotResultBlocked': 0, 'ShotResultOngoalWithNoGoal':0, 'ShotResultMissed':0, 'ShotResultOngoalWithGoal':1}

#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
#marker options: 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'
#markers = ['D','*','s','h','X','X','s','s']

from matplotlib import pyplot as plt
import matplotlib
font = {'family' : 'normal', 'size'   : 18}
matplotlib.rc('font', **font)
import matplotlib.cm as cm
plt.rcParams["figure.figsize"] = (30,18)


class ShotSnapshot():
    def __init__(self, entity_dict_snapshot):
        self.entity_dict = entity_dict_snapshot
        self.node_list = self._construct_node_list_w_attribute()
        self.edge_list= self._construct_edge_list_w_weight()
        self.G= self.build_nx_graph_from_df(self.node_list, self.edge_list)
    
    def _calculate_two_point_distance(self, p1, p2):
        #goalie_pt = np.array([goalie_loc["X"], goalie_loc["Y"]])
        #shot_pt = np.array([shot_loc["X"], shot_loc["Y"]])

        return np.linalg.norm(p1 - p2, ord=2)
    
    def _calculate_velocity(self, velocity_vector):
        return np.sqrt(velocity_vector.dot(velocity_vector))
        
    def _calculate_velocity_degree(self, velocity_vector):
        if velocity_vector[0]!=0:
            return np.arctan(velocity_vector[1]/velocity_vector[0])
        else:
            return np.pi/2
    
    def _determine_node_type(self, pos_name):
        if 'other' not in pos_name:
            return pos_name
        else:
            return pos_name.rstrip(pos_name[-1])
        
    def _construct_node_list_w_attribute(self):
        """build node list with color"""
        all_pos = list(self.entity_dict.keys())
        #print(all_pos)

        node_list = [
            (pos, 
             {
                 'color':NODE_COLOR_DICT[self._determine_node_type(pos)], 
                 'UTC': self.entity_dict[pos]['ts'],
                 'velocity': [self._calculate_velocity(np.array((velocity_vec['X'], velocity_vec['Y'])))  for velocity_vec in self.entity_dict[pos]['velocity']],
                 'degree': [self._calculate_velocity_degree(np.array((velocity_vec['X'], velocity_vec['Y'])))  for velocity_vec in self.entity_dict[pos]['velocity']],
                 'velocity_raw': [np.array((velocity_vec['X'], velocity_vec['Y']))  for velocity_vec in self.entity_dict[pos]['velocity']],
             }
            ) 
            for pos in all_pos
        ]
        return node_list
    
    def _construct_edge_list_w_weight(self):
        """build edge based on distance"""
        edge_list = []
        all_pos = list(self.entity_dict.keys())
      
        for i in range(len(all_pos)):
            for j in range(i): #only half triangle
                p1 = np.array([self.entity_dict[all_pos[i]]['location']["X"], self.entity_dict[all_pos[i]]['location']["Y"]])
                p2 = np.array([self.entity_dict[all_pos[j]]['location']["X"], self.entity_dict[all_pos[j]]['location']["Y"]])
                dist = self._calculate_two_point_distance(p1, p2)
                if ~np.isnan(dist):
                    if dist!=0:
                        edge_list.append((all_pos[i], all_pos[j], {'weight': 1/dist}))
                        #edge_list.append((all_pos[j], all_pos[i], {'weight': 1/dist}))
                    else:
                        edge_list.append((all_pos[i], all_pos[j], {'weight': 10}))
        #print(edge_list)
        #print(len(edge_list))
        return edge_list
       
    def build_nx_graph_from_df(self, node_list, edge_list):
        nx_g = nx.Graph()
        nx_g.add_nodes_from(
            node_list, #node list with node 2-tuples of the form (node, node_attribute_dict)
            #velocity=self._construct_feature_matrix(node_list), #this is broadcasted to every node 
            #feat2=np.zeros((correlation_matrix.shape[0], 1)), #add more note feature
        )
        nx_g.add_edges_from(
            edge_list, #edge list with 3-tuple with 2 nodes followed by an edge attribute dictionary, e.g., (2, 3, {'weight': 3.1415})
        )
        return nx_g
    
    
    def draw_nx_graph_at_specific_time(self, shot_id, high_threshold=0.2, low_threshold=0.1, save_figure=False):
        """only draw nodes with anomaly"""
        fig, ax = plt.subplots(1,1)

        #draw edges with different weights https://networkx.org/documentation/stable/auto_examples/drawing/plot_weighted_graph.html#sphx-glr-auto-examples-drawing-plot-weighted-graph-py
        G = self.G
        
        # nodes
        node_colors = [d['color'] for (u,d) in G.nodes(data=True)]
        #pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility
        #pos = nx.spiral_layout(G)  
        pos = nx.nx_pydot.graphviz_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors)
        edge_weight = [d['weight'] for (u, v, d) in G.edges(data=True)]
        # edges
        if len(edge_weight)>0:
      
            elarge = [(u, v) for (u, v, d) in  G.edges(data=True) if np.abs(d["weight"]) >= high_threshold]
            esmall = [(u, v) for (u, v, d) in  G.edges(data=True) if low_threshold<= np.abs(d["weight"]) < high_threshold]
            nx.draw_networkx_edges(G, pos, edgelist=elarge, width=5, alpha=0.8,)
            nx.draw_networkx_edges(G, pos, edgelist=esmall, width=5, alpha=0.5, edge_color="b", style="dashed")
        # labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif", font_color='k')

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        time = self.entity_dict['puck']['ts']
        plt.title('Shot graph at time {}'.format(time))
        plt.tight_layout()
        #plt.legend()
        if save_figure:
            fig.set_dpi(300)
            plt.savefig(f'/home/ec2-user/SageMaker/NHLShotQuality/reports/figures/entity_relation_graph_for_shot_{shot_id}_at_{time}.png' )
        else:
            plt.show()


class GameDataToGraph():
    """Convert player and puck trajectory to data ready to load into Dynamic GCN"""
    
    def __init__(self, all_game_data_dir,  graph_dir, label_dir, load_all_games=False,  snapshot_num=5, game_idx=0, use_raw_velocity_vector=True):
        """
        Create Dynamic Heterogenous Graph with edge being distances between entities  
        Arg:
            all_game_data_dir: directory of processed game
        """
        all_games_pickle_file = os.listdir(all_game_data_dir)
        all_games_pickle_file = [f for f in all_games_pickle_file if '.pickle' in f]
        #all_games_pickle_file_filtered = list(filter(lambda f: int(f.split('.')[0].split('_')[-1].strip('HITS'))%2!=0, all_games_pickle_file)) #only odd games 
        #all_games_pickle_file_filtered.sort(key=self.sort_key_for_pickle_file)
        all_games_pickle_file.sort(key=self.sort_key_for_pickle_file)
        self.make_data_dir(graph_dir)
        self.graph_path = graph_dir
        self.label_path = label_dir
        #self.game_list = []
        #for game_f in all_games_pickle_file_filtered:
        if load_all_games:
            for game_f in all_games_pickle_file:
                game = load_pickle(os.path.join(all_game_data_dir, game_f))
                print(f'loaded {game_f}')
                self.save_game_shots(game, snapshot_num=snapshot_num, use_raw_velocity_vector=use_raw_velocity_vector)
                del game
                gc.collect()
                #self.game_list.append(game)
        else:
            game_f = all_games_pickle_file[game_idx]
            game = load_pickle(os.path.join(all_game_data_dir, game_f))
            print(f'loaded {game_f}')
            self.save_game_shots(game, snapshot_num=snapshot_num, use_raw_velocity_vector=use_raw_velocity_vector)
            #self.game_list.append(game)

        
    def sort_key_for_pickle_file(self, f):
        """sort the game id by the hits sequence"""
        return int(f.split('.')[0].split('_')[-1].strip('HITS'))
        
    def make_data_dir(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            
    def save_game_shots(self, game, snapshot_num=5, use_raw_velocity_vector=True):
        id_label_dict = {}
        print('process game {}'.format(game.get_game_id()))
        try:
            matched_shots_for_game, matched_utc_for_game = game.custom_match_hits_with_ppt_utc()
        except Exception as e:
            print(e)
            print('Encounter error for game {}'.format(game.get_game_id()))
            matched_shots_for_game = None
        if matched_shots_for_game is not None:
            matched_shots_by_outcome_for_game = game.separate_shot_by_result(matched_shots_for_game)
            for outcome in matched_shots_by_outcome_for_game:
                #shot_idx = 0
                for _, ppt_shot in matched_shots_by_outcome_for_game[outcome]:
                    if ppt_shot.puck is not None and ppt_shot.goalie is not None and ppt_shot.shooter is not None and len(ppt_shot.puck.location_list)>0 and len(ppt_shot.shooter.location_list)>0 and len(ppt_shot.goalie.location_list)>0:
                        shot_id = game.get_game_id()+'_'+str(ppt_shot.timestamp)
                        print('process shot {}'.format(shot_id))
                        id_label_dict.setdefault(shot_id, outcome)
                        try:
                            self.save_snapshots(ppt_shot, shot_id, snapshot_num, outcome, use_raw_velocity_vector=use_raw_velocity_vector)
                        except Exception as e:
                            print(e)

            json_file_path= '{}/{}_outcome.json'.format(self.label_path, game.get_game_id())
            save_json_file(json_file_path, id_label_dict)
        
    def _construct_entity_graph_snapshot(self, entity_dict, snapshot_index):
        print('process shot idx {}'.format(snapshot_index))
        entity_dict_snapshot = {}
        for pos in entity_dict:
            if pos != 'other_defend_player' and pos != 'other_offend_player':
                if len(entity_dict[pos]['location'])>0:
                    entity_dict_snapshot.setdefault(pos, 
                                                    {
                                                        'location': entity_dict[pos]['location'][snapshot_index],
                                                        'velocity': entity_dict[pos]['velocity'],
                                                        'ts': entity_dict[pos]['ts'][snapshot_index],
                                                    }
                                                   )
            else:
                for i, player in enumerate(entity_dict[pos]):
                    if len(player['location'])>0:
                        entity_dict_snapshot.setdefault(pos+str(i), 
                                                        {
                                                            'location': player['location'][snapshot_index],
                                                            'velocity': player['velocity'],
                                                            'ts': player['ts'][snapshot_index],
                                                        }
                                                       )
        return entity_dict_snapshot
    
    
    def _determine_node_index_from_pos_name(self, pos, node_list):
        all_pos = [node_d[0] for node_d in node_list]
        other_offend_player = [int(p[-1]) for p in all_pos if 'other_offend_player' in p]
        other_defend_player = [int(p[-1]) for p in all_pos if 'other_defend_player' in p]
        if len(other_offend_player)>0:
            num_of_other_offend_player = max(other_offend_player)+1
        else:
            num_of_other_offend_player = 0
        if len(other_defend_player)>0:
            num_of_other_defend_player = max(other_defend_player)+1
        else:
            num_of_other_defend_player = 0
        if pos in NODE_INDEX_DICT:
            return NODE_INDEX_DICT[pos]
        else:
            if 'other_offend_player' in pos:
                pos_idx = 6+ int(pos[-1])
            elif 'other_defend_player' in pos:
                pos_idx = 6+ int(pos[-1]) + num_of_other_offend_player
            else:
                raise NotImplementedError 
            return pos_idx
        
    
    def _construct_edge_data_from_edge_list(self, edge_list, node_list):
        total_edges = len(edge_list)
        edge_index = np.zeros((2, total_edges))
        edge_weight = np.zeros((total_edges, 1))
        for i,edge in enumerate(edge_list):
            pos_1_idx = self._determine_node_index_from_pos_name(edge[0], node_list)
            pos_2_idx = self._determine_node_index_from_pos_name(edge[1], node_list)
            weight = edge[2]['weight']
            
            if pos_1_idx<NUM_ENTITIES and  pos_2_idx<NUM_ENTITIES:

                edge_index[0][i] = pos_1_idx
                edge_index[1][i] = pos_2_idx
                edge_weight[i][0] = weight
                
                #edge_matrix[pos_1_idx][pos_2_idx] = weight
                #edge_matrix[pos_2_idx][pos_1_idx] = weight
        return edge_index, edge_weight
       
    def _construct_node_feature_from_node_list(self, snapshot_num, node_list, use_raw_velocity_vector=True):

        feature_matrix = np.zeros((NUM_ENTITIES, snapshot_num*2)) #raw velocity vec


        for i,node_data in enumerate(node_list):
            pos_idx = self._determine_node_index_from_pos_name(node_data[0], node_list)
            if pos_idx < NUM_ENTITIES:
                #print('{} index is {}'.format(node_data[0], pos_idx))
                #print('Velocity vector:')
                #print(node_data[1]['velocity'])
                if use_raw_velocity_vector:
                    feature_matrix[pos_idx,:] = np.array(node_data[1]['velocity_raw']).reshape(1,-1).squeeze() #raw velocity vector
                else:
                    feature_matrix[pos_idx,:] = np.array([node_data[1]['velocity'],node_data[1]['degree']]).reshape(1,-1).squeeze()
                    
        return feature_matrix
   
    def save_snapshot_data(self, entity_dict, shot_id, snapshot_num, snapshot_index, outcome, use_raw_velocity_vector=True):
        """save graph data at each snapshot"""
        #root_index = np.array(self.root_index)
        #root_features = np.array(self.root_features)
        #indexing into the snapshot(default each other 1s during 5s before the shot release)       
        entity_dict_snapshot = self._construct_entity_graph_snapshot(entity_dict, snapshot_index)
        #print(entity_dict_snapshot)
        shot_snapshot = ShotSnapshot(entity_dict_snapshot)
        edge_list= shot_snapshot.edge_list
        node_list = shot_snapshot.node_list
        edge_index, edge_weight = self._construct_edge_data_from_edge_list(edge_list, node_list)
        feature_matrix = self._construct_node_feature_from_node_list(snapshot_num, node_list, use_raw_velocity_vector=use_raw_velocity_vector)
        label = outcome_label[outcome]
        FILE_PATH = "{}/{}_{}_{}.npz".format(
            self.graph_path, shot_id, snapshot_index, snapshot_num
        )
        np.savez(  # save snapshots
            FILE_PATH,
            x=feature_matrix,  #feature 
            y=label, #label
            edge_index=edge_index, #edge
            edge_weight = edge_weight, #edge weight
        )
                
    def save_snapshots(self, shot, shot_id, snapshot_num, outcome, use_raw_velocity_vector=True):
        """save all snapshots of a shot"""
        time_interval = 5/snapshot_num #a fixed 5s time window
        snapshot_ts = [shot.timestamp-5+(i+1)*time_interval for i in range(snapshot_num)] 
        #snapshot_ts = [shot.timestamp-snapshot_num+1+i for i in range(snapshot_num)] 
        shooter_team = shot.shooter.player_entity_registration['VisOrHome']
        #print(shot.puck)
        #print(shot.puck.location_list)
        entity_dict = {
            'puck' : {
                'ts': snapshot_ts,
                'location' : [shot.puck.location_list[shot.puck._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if shot.puck is not None and len(shot.puck.location_list)>0],
                'velocity' : [shot.puck.velocity_list[shot.puck._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if shot.puck is not None and len(shot.puck.velocity_list)>0],
            },
            'shooter' : {
                'ts': snapshot_ts,
                'location' : [shot.shooter.location_list[shot.shooter._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if shot.shooter is not None and len(shot.shooter.location_list)>0],
                'velocity' : [shot.shooter.velocity_list[shot.shooter._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if shot.shooter is not None and len(shot.shooter.velocity_list)>0],
            },
            'closest_defender' : {
                'ts': snapshot_ts,
                'location' : [shot.closest_defender.location_list[shot.closest_defender._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if shot.closest_defender is not None and len(shot.closest_defender.location_list)>0],
                'velocity' : [shot.closest_defender.velocity_list[shot.closest_defender._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if shot.closest_defender is not None and len(shot.closest_defender.velocity_list)>0],                
            },
            'goalie': {
                'ts': snapshot_ts,
                'location' : [shot.goalie.location_list[shot.goalie._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if shot.goalie is not None and len(shot.goalie.location_list)>0],
                'velocity' : [shot.goalie.velocity_list[shot.goalie._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if shot.goalie is not None and len(shot.goalie.velocity_list)>0],                
            }, 
            'shot_assister':{
                'ts': snapshot_ts,
                'location' : [shot.shot_assister.location_list[shot.shot_assister._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if shot.shot_assister is not None and len(shot.shot_assister.location_list)>0],
                'velocity' : [shot.shot_assister.velocity_list[shot.shot_assister._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if shot.shot_assister is not None and len(shot.shot_assister.velocity_list)>0],                
            }, 
            'non_goalie_shot_blocker':{
                'ts': snapshot_ts,
                'location' : [shot.non_goalie_shot_blocker.location_list[shot.non_goalie_shot_blocker._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if shot.non_goalie_shot_blocker is not None and len(shot.non_goalie_shot_blocker.location_list)>0],
                'velocity' : [shot.non_goalie_shot_blocker.velocity_list[shot.non_goalie_shot_blocker._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if shot.non_goalie_shot_blocker is not None and len(shot.non_goalie_shot_blocker.velocity_list)>0],                
            },
            'other_offend_player':[
                {
                    'ts': snapshot_ts,
                    'location' : [player.location_list[player._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if player is not None and len(player.location_list)>0],
                    'velocity' : [player.velocity_list[player._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if player is not None and len(player.velocity_list)>0],
                }
                for player in shot.other_players if player.player_entity_registration['VisOrHome'] == shooter_team
            ], 
            'other_defend_player':[
                {
                    'ts': snapshot_ts,
                    'location' : [player.location_list[player._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if player is not None and len(player.location_list)>0],
                    'velocity' : [player.velocity_list[player._find_closest_tracking_msg_idx(ts)] for ts in snapshot_ts if player is not None and len(player.velocity_list)>0],
                }
                for player in shot.other_players if player.player_entity_registration['VisOrHome'] != shooter_team
            ],
        }
        FILE_PATH = "{}/{}_5s_snapshots.json".format(self.graph_path, shot_id)
        save_json_file(FILE_PATH, entity_dict) 
        for i in range(snapshot_num):
            self.save_snapshot_data(entity_dict, shot_id, snapshot_num, i, outcome, use_raw_velocity_vector=use_raw_velocity_vector)
        return entity_dict
        
    def plot_node_feature_at_specific_time(self, shot_id, snapshot_num, snapshot_index,):
        pass
            

    def save_dynamic_graph_visual(self, shot_id, snapshot_num, high_threshold=0.2, low_threshold=0.1, save_figure=True):

        FILE_PATH = "{}/{}_5s_snapshots.json".format(
            self.graph_path, shot_id
        )
        entity_dict = load_json_file(FILE_PATH)
        for i in range(snapshot_num):
            entity_dict_snapshot = self._construct_entity_graph_snapshot(entity_dict, i)
            shot_snapshot = ShotSnapshot(entity_dict_snapshot)
            shot_snapshot.draw_nx_graph_at_specific_time(shot_id, high_threshold=high_threshold, low_threshold=low_threshold, save_figure=save_figure)

    

def main():
    # -------------------------------
    #         PARSE ARGUMENTS
    # -------------------------------
    arg_names = ['command', 'snapshot_num', 'use_raw_velocity_vector']
    if len(sys.argv) != 3:
        print("Please check the arguments.\n")
        print("Example usage:")
        print("python graph_data_preprocess.py  5 0 \n") #0 is false, 1 is true
        exit()
    args = dict(zip(arg_names, sys.argv))
    snapshot_num, use_raw_velocity_vector = int(args['snapshot_num']), int(args['use_raw_velocity_vector'])
    #snapshot_num, use_raw_velocity_vector = int(args['snapshot_num']), args['use_raw_velocity_vector']
    print_dict(args)
    all_game_data_dir = '/home/ec2-user/SageMaker/NHLShotQuality/data/processed/'
    #print(type(use_raw_velocity_vector))
    
    if use_raw_velocity_vector:
        graph_dir = f'/home/ec2-user/SageMaker/NHLShotQuality/data/graph_raw_velocity_vector_{snapshot_num}_snapshot/'
    else:
        graph_dir = f'/home/ec2-user/SageMaker/NHLShotQuality/data/graph_{snapshot_num}_snapshot/'
    print(f"save processed graph data into {graph_dir}")
    label_dir = '/home/ec2-user/SageMaker/NHLShotQuality/data/label/'
    game_graph_data = GameDataToGraph( all_game_data_dir,  graph_dir, label_dir, load_all_games=True, snapshot_num=snapshot_num, use_raw_velocity_vector=use_raw_velocity_vector)
    #game_graph_data = GameDataToGraph( all_game_data_dir,  graph_dir, load_all_games=False, snapshot_num=5, game_idx=193)

if __name__ == '__main__':
    start_time = time.time()  # Timer Start
    main()
    end_time = time.time()
    print("\nElapsed Time: {0} seconds".format(
        round(end_time - start_time, 3)))
