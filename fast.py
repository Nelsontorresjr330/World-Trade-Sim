# Standard Libraries
from dataclasses import dataclass, field
import os, re, csv, math, copy, random, uuid, time, multiprocessing, keyboard
from typing import List

#Global Constants
k = 0.1 #The close to zero, the more likely of schedule success
x_0 = 1 #This is basically a paperweight, x_0 is constantly reassigned
gamma = 0.99 #Must be between 0 and 1, the higher the gamma, the less the penalty for DR
C = -0.01 #Cost of failing a schedule, Negative Constant, the further from zero, the greater the penalty

#All logic involving the template and the parser is thanks to the piazza post, 
#I'm not sure how to see the students name but I greatly appreciate them!
@dataclass
class ResourceQuantity:
 name: str = field()
 quantity: int = field()

@dataclass
class TransformTemplate:
 name: str = field(default="")
 inputs: List[ResourceQuantity] = field(default_factory=list)
 outputs: List[ResourceQuantity] = field(default_factory=list)
 def __str__(self) -> str:
    return_string = f"(TRANSFORM {self.name.upper()} (INPUTS {str(self.inputs).upper()}) (OUTPUTS {str(self.outputs).upper()})"     
    return return_string

def read_file(file_path: str) -> List[dict]:
 file_contents = None
 with open(file_path, mode='r') as file:
    file_contents = file.read()
 return file_contents

def validate_nonempty(template: str = "") -> bool:
 return template != ""

def validate_enclosed(template: str = "") -> bool:
 left_paren_count = template.count("(")
 right_paren_count = template.count(")")
 if left_paren_count != right_paren_count:
    return False
 return True

def validate_keywords(template: str = "") -> bool:
 transform_keywords = ["TRANSFORM", "INPUTS", "OUTPUTS"]
 for keyword in transform_keywords:
    if not keyword in template:
        return False
 return True

def validate(template: str = ""):
 if not validate_nonempty(template):
  raise Exception("Empty template")
 if not validate_enclosed(template):
  raise Exception("Incorrect parentheses counts, verify all expressions are properly enclosed")
 elif not validate_keywords(template):
  raise Exception("Missing required keywords, verify transform is syntactically correct")

def build_resource_quantities(resource_quantities_block):
 quantities = []

 regex = r"\(([A-Za-z]+) (\d)\)"
 matches = re.finditer(regex, resource_quantities_block, re.MULTILINE)
 for match in matches:
   resource_name, resource_quantity = match.groups()
   quantities.append(ResourceQuantity(name=resource_name, quantity=int(resource_quantity)))

 return quantities

def build_transform_template(template_path: str, template: str) -> TransformTemplate:
 transform = TransformTemplate()
 
 basename = os.path.basename(template_path)
 transform_name = os.path.splitext(basename)[0]
 transform.name = transform_name

 inputs_start = template.index("INPUTS")
 outputs_start = template.index("OUTPUTS")

 inputs_string = template[inputs_start:outputs_start]
 outputs_string = template[outputs_start:]

 transform.inputs = build_resource_quantities(inputs_string)
 transform.outputs = build_resource_quantities(outputs_string)

 return transform

def parse(template_path: str) -> TransformTemplate:
 template = read_file(template_path)
 validate(template)
 transform_template = build_transform_template(template_path, template)
 return transform_template

'''I greatly simplified my SQ function to improve performance & also to see how the AI adapts to it.
The new logic is still based on the same "focus on needs, then the rest" logic as before but much
less human intuition. It just doubles the points given from any "need" if the need is 1.25X the population.'''
def state_quality(country):
    #Variables
    resources = country.resources
    weights = country.weights
    population = resources["Population"]
    total_quality = 0.0

    #Check for need bonus
    for resource in available_resources:
        #If the resource is a need
        if resource in ["Housing","Food","Water"]:
            #If there is less of the resource than there is population, do nothing
            if resources[resource] <= population:
                break
            #If there is an excess of needs, grant the bonus (2X the points granted from the resource)
            elif resources[resource] >= (population*1.25):
                total_quality += resources[resource]
                
    #Calculate state quality using resources and weights
    for resource in available_resources:
        resource_val = float(resources[resource])
        weight_val = float(weights[resource])

        resource_points = resource_val*weight_val
        total_quality += resource_points

    return total_quality

def undiscounted_reward(country):
    UR = (country.load_initial_quality() - country.get_current_quality())
    return UR
   
#Gamma must be between 0.0-1.0
def discounted_reward(country):
    normalizer = gamma ** country.steps
    DR = normalizer * country.get_undiscounted_reward()
    return DR 

#https://en.wikipedia.org/wiki/Logistic_function
def country_accept_prob(country, x0 = x_0):
    try:
        exponential = ((-k)*(country.get_discounted_reward()-x0))
        denominator = 1.0 + (math.e ** float(exponential))
    except Exception as e:
       print(f"Overflow Error, exponential = {exponential} \n{e}")
       exit()
    return (1/denominator)
   
#Not the summation of all success rates, the product of all success rates
def schedule_success_prob(x0 = x_0):
    success_rate = 1
    for country in world_states:
        success_rate = success_rate * country_accept_prob(world_states[country],x0)
    return success_rate

def expected_utility(country, x0 = x_0):
    EU = schedule_success_prob(x0) * country.get_discounted_reward() + ((1 - schedule_success_prob(x0)) * C)
    return EU
   
#Main state, grants UUID upon creation :D
class state:
    resources = {}
    weights = {}
    transformations = []                #HEY FIX ME & MY OUTPUT
    initial_quality = None
    current_quality = None
    undiscounted_reward = None
    discounted_reward = None
    expected_utility = None
    steps = 1

    def __init__(self, name) -> None:
        self.name = name
    
    def load_initial_quality(self):
        if self.initial_quality is None:
           self.initial_quality = state_quality(self)
        return self.initial_quality

    def get_undiscounted_reward(self):
        self.undiscounted_reward = undiscounted_reward(self)
        return self.undiscounted_reward
    
    def get_current_quality(self):
        self.current_quality = state_quality(self)
        return self.current_quality
    
    def get_discounted_reward(self):
        self.discounted_reward = discounted_reward(self)
        return self.discounted_reward

    def get_expected_utility(self, x0 = x_0):
       self.expected_utility = expected_utility(self, x0)
       return self.expected_utility

    #Used to make sure all values are floats and able to be operated on
    def change_to_floats(self):
       for resource in self.resources:
          if resource != "Country":
             self.resources[resource] = float(self.resources[resource])

    #Used in the initialization
    def load_weights(self, weight_file_path):
        #Variables
        weights_dict = {}

        #Open & read the input file into the dict
        with open(weight_file_path, encoding='utf-8') as csv_file_handler:
            csv_reader = csv.DictReader(csv_file_handler)

            for rows in csv_reader:
               key = rows['Resource']
               weights_dict[key] = rows
        
        #Get its values
        list_of_weights = []
        for val in weights_dict:
           list_of_weights.append(weights_dict[val])

        #Put values into the states weights
        for temp in list_of_weights:
           self.weights[temp['Resource']] = temp['Weight']

#Used to read in all the values from the input
def initialize_world(initial_world_path):
    #Variables
    states_dict = {}
    states_resources_dict = {}

    #Read in all the values from CSV
    with open(initial_world_path, encoding='utf-8') as csv_file_handler:
        csv_reader = csv.DictReader(csv_file_handler)

        for rows in csv_reader:
           key = rows['Country']
           states_resources_dict[key] = rows

        temp = list(list(states_resources_dict.values())[0].keys()) #Fancy logic to get all the resources
        del temp[0]

    #Global to allow for logic to be done outside of function
    global available_resources
    available_resources = temp

    #Get just the values
    states_resources_list = list(states_resources_dict.values())

    #Initialize states with their names
    for country in states_resources_list:
        states_dict[country["Country"]] = state(country["Country"])

    #Assign each country their resources
    for country in states_dict:
       if states_dict[country].name in states_resources_dict.keys():
          states_dict[country].resources = states_resources_dict[country]

    #Convert each countries resources to floats
    for country in states_dict:
       states_dict[country].change_to_floats()

    return states_dict

#Process for transformations
def run_transformation(transfer, state, multiplier):

    #For each input in the tranformation
    for input in transfer.inputs:
        if state.resources[input.name] < (input.quantity*multiplier):               #If not enough resources to carry out the transformation
            # print(f"{state.name} only has {state.resources[input.name]} {input.name}, the required amount is {input.quantity*multiplier} {input.name}")
            state.steps += 1                                                        #To make sure the steps in the state get incremented
            return state
        else:
            state.resources[input.name] -= (input.quantity*multiplier)              #Remove their inputs
    
    #For each output in the transformation
    for output in transfer.outputs:
            state.resources[output.name] += (output.quantity*multiplier)            #Add their outputs

    #State values are fixed
    state.steps += 1

    return state
    
#Process for transfers
def run_transfer(country_a, direction, country_b, resource_type, amount):

    #Assign directions
    if direction.lower() == "to":
       to_country = country_b
       from_country = country_a
    elif direction.lower() == "from":
       to_country = country_a
       from_country = country_b

    #If the from country does not have the necessary resources
    if from_country.resources[resource_type] < amount:
       from_country.steps += 1                              #Increment the steps
       to_country.steps += 1
       return country_a                                     #Exit

    #Write the exit string
    review_string = f"(TRANSFER {from_country.name.upper()} {to_country.name.upper()} (({resource_type.capitalize()} {amount})))"

    #Assign values
    from_country.resources[resource_type] -= amount
    to_country.resources[resource_type] += amount

    #Increment steps
    from_country.steps += 1
    to_country.steps += 1

    #Include exit string and return
    country_a.transformations.append(review_string + '\n')

    return country_a

#Node class for Heuristic DFS
class node:
    def __init__(self, parent, id, EU, transformation, state) -> None:
        self.parent = parent
        self.id = id
        self.EU = EU
        self.transformation = transformation
        self.state = state

    #Used for printing and debugging
    def __str__(self) -> str:
       return f"ID : {self.id} | EU : {self.EU} | Transformation : {self.transformation}"

#Frontier class for Heuristic DFS
class frontier:
    def __init__(self, parent_id, nodes, depth) -> None:
        self.id = str(uuid.uuid4())
        self.parent_id = parent_id
        self.nodes = nodes  
        self.depth = depth

    #Used for printing and debugging
    def __str__(self) -> str:
       return f"Parent : {self.parent_id} | {self.depth}\n Nodes :{self.nodes}"

#Depth First Search, Multicore based on Expected Utility
def dfs(depth_bound, max_frontier, country, transformations, max_multiplier = 2):

    #Variables
    # global graph  #Contains all the frontiers w/ all the nodes
    graph = {}
    possible_transformations = []   #List of all possible moves
    visited = []    #List of all node that have been visited 
    frontiers = []  #List of all frontiers
    first_nodes = []    #List of all nodes in the initial frontier
    parent_id = None    #Used to backtrack graph
    iter = 0    #Used to get all possible transformations
    layer = 1   #Used to keep track of frontier layers

    #Set actions based on max_frontier
    #Start with all possible transformations
    for iter, transformation in enumerate(transformations):
        if iter >= len(transformations):
            break
        else:
            possible_transformations.append([transformations[transformation], random.randrange(1,max_multiplier)])

    #Give iteration that breaks the loop
    iter += 1

    #List of countries to for possible transformation
    country_names = list(world_states.keys())   
    country_names.remove(country.name)

    #Include Random Transfers
    while iter < max_frontier:
        possible_transformations.append([random.choice(available_resources), random.choice(["To","From"]), random.choice(country_names), random.randrange(1,max_multiplier)])

        #Delete any dupes
        res = []
        [res.append(x) for x in possible_transformations if x not in res]
        possible_transformations = res

        #Iterate (assuming it wasnt a dupe)
        iter = len(possible_transformations)

    #Get initial x_0 value
    discounted_rewards = []
    for each_country in country_names:
        discounted_rewards.append(discounted_reward(world_states[each_country]))
    avg_dr = sum(discounted_rewards)/len(discounted_rewards)

    #Create the first frontier
    #Add all SQs on the frontier
    for iter, choice in enumerate(possible_transformations):
        #Create node for frontier
        node_id = str(uuid.uuid4())
        dupe = copy.copy(country)

        #Run transformation, save its new SQ as its value under its UUID
        if(len(choice) == 2):
            modded_dupe = run_transformation(choice[0],dupe,choice[1])  
        else:
            modded_dupe = run_transfer(dupe, choice[1], world_states[choice[2]], choice[0], choice[3]) 

        #Create the node with all its values & add it to the all nodes dict
        first_nodes.append(node(parent_id, node_id, modded_dupe.get_expected_utility(avg_dr),choice,modded_dupe))

    #Create the first frontier object & add it to the graph
    first_frontier = frontier(parent_id, first_nodes, 0)
    frontiers.append(first_frontier)

    #Calculate the initial x_0 value, as part of my part 2 changes,
    #I was curious to see how a dynamic x_0 value affected the AI's
    #Ability to increase its EU
    discounted_rewards = []
    for each_country in country_names:
        discounted_rewards.append(discounted_reward(world_states[each_country]))
    avg_dr = sum(discounted_rewards)/len(discounted_rewards)    #avg_dr is the new x_0

    initial_frontiers = []

    for nodes_inside in first_frontier.nodes:
        node_to_visit = nodes_inside
        parent_id = node_to_visit.id                #Assign the new parent id
        current_nodes = []                          #Prepare list

        #Add all SQs on the frontier
        for iter, choice in enumerate(possible_transformations):
            node_id = str(uuid.uuid4())             #Create nodes for frontier
            dupe = copy.copy(node_to_visit.state)   #Each node is a copy of the one meant to visit

            #Run transformation, save its new SQ as its value under its UUID
            if(len(choice) == 2):
                modded_dupe = run_transformation(choice[0],dupe,choice[1])  
            else:
                modded_dupe = run_transfer(dupe, choice[1], world_states[choice[2]], choice[0], choice[3]) 

            #Create & Add the node to the list
            current_nodes.append(node(parent_id, node_id, modded_dupe.get_expected_utility(avg_dr),choice,modded_dupe))
            
        #Create a new frontier with each new node
        initial_frontiers.append(frontier(parent_id, current_nodes, layer))

    #Parallel Processing Logic
    #Fork must be used to ensure global variables work properly across children
    multiprocessing.set_start_method('fork')

    #Manager.list used to keep the outputs of each scheduler
    manager = multiprocessing.Manager()
    return_list = manager.list()

    #Variable used to keep track of processes
    jobs = []

    for iter in range(len(initial_frontiers)):
        graph = {}
        graph[0] = frontier(None,[first_nodes[iter]],0)
        graph[1] = initial_frontiers[iter]

        visited = []

        # discounted_rewards = []
        # for each_country in country_names:
        #     discounted_rewards.append(discounted_reward(world_states[each_country]))
        # avg_dr = sum(discounted_rewards)/len(discounted_rewards)    #avg_dr is the new x_0

        #Spawn a process and save it under jobs
        p = multiprocessing.Process(target=evaluate_path,args=[graph[0], avg_dr, depth_bound - 1, graph, visited, possible_transformations, frontiers, country_names, return_list])
        p.start()
        jobs.append(p)

    #Once all processes are spawned, wait for all of them to finish & close them
    for proc in jobs:
        proc.join()

    # for best_nodes in return_list:
    #    print(best_nodes)

    return

    print(true_best_node, '\n','\n')
    print(graph, '\n','\n')

    action_list = []
    iter_node = true_best_node

    action_list.append(f"Best EU: {true_best_node.EU}")

    while iter_node.parent is not None:
       action_list.append(iter_node.transformation)
       for visited_node in visited:
          if visited_node.id == iter_node.parent:
             iter_node = visited_node
             break

    action_list.append(iter_node.transformation)
    for visited_node in visited:
      if visited_node.id == iter_node.parent:
        iter_node = visited_node
        break

    return action_list

def evaluate_path(first_frontier, avg_dr, depth_bound, graph, visited, possible_transformations, frontiers, country_names, output_list):

    #Assign initial values
    current_frontier = first_frontier
    layer = 0
    out_of_nodes = False
    true_best_node = node(None, None, -1000.0, None, None)

    #Final break when initial frontier is all visited
    while not out_of_nodes:
        #Loops through until max layer is reached
        while layer < depth_bound:
            highest_EU = -10000
            node_to_visit = None

            #If not the initial frontier
            if(layer != 0):
                for current_node in graph.get(layer-1).nodes:   #For each node
                    if current_node not in visited:             #If the node was NOT visited before
                        if (current_node.EU > highest_EU):      #If it contains a higher EU than the highest EU
                            highest_EU = current_node.EU        #Reassign the highest EU
                            node_to_visit = current_node
            else:
                for current_node in graph.get(layer).nodes:
                    if current_node not in visited:
                        if (current_node.EU > highest_EU):
                            highest_EU = current_node.EU
                            node_to_visit = current_node

            #If all nodes were previously checked
            if highest_EU == -10000:
                if layer == 0:                                  #If in the initial layer
                   out_of_nodes = True                          #Assign conditional and break
                   break 
                layer -= 1                                      #Return one layer

            #If all nodes were not checked in this layer
            else:
                parent_id = node_to_visit.id                    #Assign the new parent id

                if layer != depth_bound:                        #If we are not at the max layer
                    current_nodes = []                          #Prepare list

                    #Add all SQs on the frontier
                    for iter, choice in enumerate(possible_transformations):
                        node_id = str(uuid.uuid4())             #Create nodes for frontier
                        dupe = copy.copy(node_to_visit.state)   #Each node is a copy of the one meant to visit

                        #Run transformation, save its new SQ as its value under its UUID
                        if(len(choice) == 2):
                            modded_dupe = run_transformation(choice[0],dupe,choice[1])  
                        else:
                            modded_dupe = run_transfer(dupe, choice[1], world_states[choice[2]], choice[0], choice[3]) 

                        #Create & Add the node to the list
                        current_nodes.append(node(parent_id, node_id, modded_dupe.get_expected_utility(avg_dr),choice,modded_dupe))
                        
                    #Create a new frontier with each new node
                    current_frontier = frontier(parent_id, current_nodes, layer)

                    #For each node in this frontier, if its EU is higher than the current best choice
                    for each_node in current_frontier.nodes:
                       if each_node.EU > true_best_node.EU:
                          true_best_node = each_node            #Assign the new best node

                    #Add the frontier & node to the lists and graphs
                    frontiers.append(current_frontier)
                    graph[layer] = current_frontier
                    visited.append(node_to_visit)

                    #Write the output message
                    # txt = "Elapsed Time = {:<5} seconds : Visited {:<5} nodes with the highest possible EU of {:<6}. To exit, press Q"
                    # print(txt.format(round((time.time()-start_time), 1),len(visited),true_best_node.EU), end='\r')

                    #Get next x_0 value
                    discounted_rewards = []
                    for each_country in country_names:
                        discounted_rewards.append(discounted_reward(world_states[each_country]))
                    avg_dr = sum(discounted_rewards)/len(discounted_rewards)

                    # if keyboard.is_pressed('a'):
                    #    return true_best_node.state.transformation

                    #Enter next Layer
                    layer += 1

        layer -= 1

    # print(true_best_node.EU)
    print(len(visited))
    output_list.append(len(visited))


def country_scheduler(your_country_name, resources_filename, initial_state_filename, output_schedule_filename, depth_bound, frontier_max_size, max_multiplier):
    possible_transformations = {}
    #CHANGE TO RELATIVE PATHS
    template_paths = ["Inputs/housing.tmpl",
                      "Inputs/electronics.tmpl",
                      "Inputs/alloys.tmpl",
                      "Inputs/farm.tmpl"]    
    #Get all possible transformations          
    for path in template_paths:
        possible_transformations[path[:-5]] = parse(path)

    #Create the initial world states
    global world_states
    world_states = initialize_world(initial_state_filename)
    world_states[your_country_name].load_weights(resources_filename)
    world_states[your_country_name].load_initial_quality()

    #Run the search
    # output = dfs(depth_bound,frontier_max_size,world_states[your_country_name],possible_transformations,max_multiplier)
    dfs(depth_bound,frontier_max_size,world_states[your_country_name],possible_transformations,max_multiplier)

    #Print the results to the file
    # output_file = open(output_schedule_filename, "w")
    # for action in output:
    #     output_file.writelines(str(action))
    #     output_file.writelines('\n')
    # output_file.close()

    print(f"Final schedule(s) writen to {output_schedule_filename}")

def main():
    #CHANGE TO RELATIVE PATHS
    initial_world_path = "Inputs/Initial_World.csv"
    weights_path = "Inputs/Resources_Weights.csv"
    output_path = "Outputs/results"

    #Initial Inputs
    depth_bound = input("\nMax Depth Bound (Must be greater than 2): ")
    while not depth_bound.isdigit():
       depth_bound = input("Max Depth Bound (Must be greater than 2): ")
    max_frontier_size = input("Max Frontier Size (Must be greater than no. of templates): ")
    while not max_frontier_size.isdigit():
       max_frontier_size = input("Max Frontier Size (Must be greater than no. of templates): ")
    max_multiplier = input("Max Multiplier Value (Must be greater than 1): ")
    while not max_multiplier.isdigit():
       max_multiplier = input("Max Multiplier Value (Must be greater than 1): ")

    #Input Processing
    if int(max_multiplier) <= 1:
       max_multiplier = 2
    if int(depth_bound) <= 1:
       depth_bound = 2
    if int(max_frontier_size) < 1:
       max_frontier_size = 4

    start_time = time.time()
    print(f"Start time = {time.ctime(start_time)}\n")

    country_scheduler("Demacia", weights_path, initial_world_path, f"{output_path}_fast.txt", int(depth_bound), int(max_frontier_size), int(max_multiplier))
    
    end_time = time.time()
    print(f"\nEnd time = {time.ctime(end_time)} \nDelta = {end_time-start_time} seconds\n")
    
if __name__ =="__main__":
    global start_time
    main()
    