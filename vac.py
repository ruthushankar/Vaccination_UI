import random
import copy
import streamlit as st
import numpy as np
import pandas as pd
import Vaccination_policy

class Agent():
    def __init__(self,state,index):
        self.state=state
        self.index=index
        self.neighbours=[]
        self.restrict=False
        self.vaccination_state=None

        self.vaccination_hist=[]

class RandomGraph():
    def __init__(self,n,p,connected=True):
        self.n=n
        self.p=p
        self.connected=connected

        self.adj_list=[]
        for i in range(n):
            self.adj_list.append([])

        for i in range(n):
            for j in range(i+1,n):
                if random.random()<p:
                    self.adj_list[i].append(j)
                    self.adj_list[j].append(i)

        if self.connected:
            for i in range(n):
                if self.adj_list[i]==[]:
                    allowed_values = list(range(0, n))
                    allowed_values.remove(i)
                    j = random.choice(allowed_values)
                    self.adj_list[i].append(j)
                    self.adj_list[j].append(i)

        dsum=0
        for i in range(n):
            dsum+=len(self.adj_list[i])
        self.average_degree=2*dsum/n

class Simulate():
    def __init__(self,graph_obj,agents,transmission_probability,num_agents_per_step,available_vaccines):
        self.graph_obj=graph_obj
        self.agents=agents
        self.transmission_probability=transmission_probability
        # self.lockdown_list=lockdown_list
        self.state_list={}
        self.state_history={}
        self.quarantine_list=[]
        self.quarantine_history=[]


        # Vaccination Policy parameters

        self.num_agents_per_step = num_agents_per_step
        self.available_vaccines=available_vaccines


        for state in transmission_probability.keys():
            self.state_list[state]=[]
            self.state_history[state]=[]

        for agent in self.agents:
            self.state_list[agent.state].append(agent.index)

        self.update()
        total_vaccine_cost=self.init_vaccinating()

    def update(self):
        for state in self.state_history.keys():
            self.state_history[state].append(len(self.state_list[state]))
        self.quarantine_history.append(len(self.quarantine_list))

    def init_vaccinating(self):
        self.vaccinating_policy = Vaccination_policy.Vaccination_policy(lambda x:self.num_agents_per_step)
        # x=0
        for vaccine in self.available_vaccines.keys():
            self.vaccinating_policy.add_vaccination(vaccine,self.available_vaccines[vaccine]['cost'],
                                                    self.available_vaccines[vaccine]['decay_days'],self.available_vaccines[vaccine]['efficacy'],
                                                    self.available_vaccines[vaccine]['capacity'])
        #     x+=x
        #
        # return x
        #     cost=self.vaccinating_policy.check_count()
        # return cost

    def simulate_days(self,days):
        for day in range(days):

            self.simulate_day(day)

    def simulate_day(self,day):
        self.reset()
        self.enact_policy(day)
        self.spread(day)
        self.update()

    def reset(self):
        for agent in self.agents:
            agent.restrict=False


    def enact_policy(self,day):
        total_cost=self.vaccinating_policy.enact_policy(day,self.agents)

        return total_cost

    def spread(self,day):
        #Inf : Sus ->2bExp
        to_beExposed=[]
        # if not self.lockdown_list[day]:
        for agent_indx in self.state_list['Susceptible']:
            agent=self.agents[agent_indx]
            p_inf=self.transmission_probability['Susceptible']['Exposed'](agent,agent.neighbours)
            if random.random()<p_inf:
                to_beExposed.append(agent_indx)
        # Inf ->Rec
        for agent_indx in self.state_list['Infected']:
            agent=self.agents[agent_indx]
            p=self.transmission_probability['Infected']['Recovered'](agent,agent.neighbours)
            self.convert_state(agent,'Recovered',p)

        # Exp -> Inf
        for agent_indx in self.state_list['Exposed']:
            agent=self.agents[agent_indx]
            p1=self.transmission_probability['Exposed']['Infected'](agent,agent.neighbours)
            # r=random.random()
            # if r<p1:
            self.convert_state(agent,'Infected',p1)

        #2bExp->Exp
        for agent_indx in to_beExposed:
            agent=self.agents[agent_indx]
            self.convert_state(agent,'Exposed',1)

    def convert_state(self,agent,new_state,p):
        if random.random()<p:
            self.state_list[agent.state].remove(agent.index)
            agent.state=new_state
            self.state_list[agent.state].append(agent.index)

def world(n,p,inf_per,days,graph_obj,beta,mu,gamma,delta,num_agents_per_step,available_vaccines):
    #print("Average degree : ",graph_obj.average_degree)
    # num_agents_per_step,available_vaccines
    agents=[]
    for i in range(n):
        state='Susceptible'
        if random.random()<inf_per:
            state='Infected'
        agent=Agent(state,i)
        agents.append(agent)

    for indx,agent in enumerate(agents):
        agent.index=indx
        for j in graph_obj.adj_list[indx]:
            agent.neighbours.append(agents[j])

    individual_types=['Susceptible','Exposed','Infected','Recovered']

    def p_infection(p_inf):  # probability of infectiong neighbour
        def p_fn(my_agent,neighbour_agents):
            p_not_inf=1
            for nbr_agent in neighbour_agents:
                if nbr_agent.state=='Infected' and agent.restrict is False:
                    p_not_inf*=(1-p_inf)
            return 1 - p_not_inf
        return p_fn


    def p_standard(p):
        def p_fn(my_agent,neighbour_agents):
            return p
        return p_fn

    transmission_prob={}
    for t in individual_types:
        transmission_prob[t]={}

    for t1 in individual_types:
        for t2 in individual_types:
            transmission_prob[t1][t2]=p_standard(0)
    transmission_prob['Susceptible']['Exposed']= p_infection(beta)
    transmission_prob['Exposed']['Infected']= p_standard(mu)
    transmission_prob['Infected']['Recovered']= p_standard(gamma)
    transmission_prob['Recovered']['Susceptible']= p_standard(delta)

    sim_obj=Simulate(graph_obj,agents,transmission_prob,num_agents_per_step,available_vaccines)
    sim_obj.simulate_days(days)
    # sim_obj.
    return sim_obj.state_history

def average(tdict,number):

    for k in tdict.keys():
        l=tdict[k]
        for i in range(len(l)):
            tdict[k][i]/=number

    return tdict

def worlds(number,n,p,inf_per,days,beta,mu,gamma,delta,num_agents_per_step,available_vaccines):

    cum_inf=0
    # cum_ld=0
    individual_types=['Susceptible','Exposed','Infected','Recovered']
    tdict={}

    for state in individual_types:
        tdict[state]=[0]*(days+1)

    for i in range(number):
        graph_obj = RandomGraph(n,p,True)
        # sim_obj2=
        # sdict = world(n,p,inf_per,days,graph_obj,beta,mu,gamma,delta,lockdown_list)
        sdict = world(n,p,inf_per,days,graph_obj,beta,mu,gamma,delta,num_agents_per_step,available_vaccines)

        for state in individual_types:
            for j in range(len(tdict[state])):
                tdict[state][j]+=sdict[state][j]

    tdict=average(tdict,number)
    values=[]
    keys=individual_types

    for i in range(len(tdict['Susceptible'])):
        values.append([])
        for state in individual_types:
            values[-1].append(tdict[state][i])

    chart_data = pd.DataFrame(
    values,
    columns=keys)
    st.line_chart(chart_data)

    for i in range(days):
        cum_inf+=tdict['Infected'][i+1]

    # for i in lockdown_list:
    #     if i:
    #         cum_ld+=n

    return cum_inf

def main():
    random.seed(0)
    st.write("""
    # Vaccination Policy
    A vaccination policy in Episimmer is a user defined intervention policy being able to cater to various vaccination strategies.
    We have designed this application as a simple point solution to Vaccination policy template of the main simulator - [Episimmer](https://github.com/healthbadge/episimmer).
    We are experimenting with SEIRS model on a G(n,p) random graph.
    Use sidebar to vary parameters
    """)
    st.write("------------------------------------------------------------------------------------")


    st.sidebar.write("World parameters")
    n=st.sidebar.slider("Number of agents", min_value=0 , max_value=5000 , value=1000 , step=100 , format=None , key=None )
    p=st.sidebar.slider("Probability(p) of an edge in G(n,p) random graph", min_value=0.0 , max_value=1.0 , value=0.01 , step=0.01 , format=None , key=None )
    p_range=st.sidebar.checkbox("Divide p by 10")

    if p_range:
        p/=10

    inf_per=0.01
    days=st.sidebar.slider("Number of days in simulation", min_value=1 , max_value=100 , value=30 , step=1 , format=None , key=None )
    num_worlds=st.sidebar.slider("Number of times to average simulations over", min_value=1 , max_value=100 , value=1 , step=1 , format=None , key=None )
    st.sidebar.write("Averaging simulation "+str(num_worlds)+" times over graph G("+str(n)+","+str(p)+") for "+str(days)+" days.")

    st.sidebar.write("------------------------------------------------------------------------------------")

    available_vaccines={}
    st.sidebar.write("Vaccination parameters")
    num_agents_per_step=st.sidebar.slider("Number of Agents to vaccinate every day", min_value=0 , max_value=1000 , value=100 , step=10 , format=None , key=None )
    num_vaccine_types=st.sidebar.slider("Number of distinct vaccines", min_value=0 , max_value=10 , value=1 , step=1 , format=None , key=None)

    tc=0
    for i in range(num_vaccine_types):
        available_vaccines['Vaccine'+str(i+1)] = {}
        st.sidebar.text("Vaccine Type {0}".format(i+1))
        cost = st.sidebar.slider("Cost of each vaccine", min_value=1 , max_value=1000 , value=1 , step=1, key=i)
        decay = st.sidebar.slider("Decay days of each vaccine", min_value=1 , max_value=100 , value=1 , step=1, key=i)
        efficacy = st.sidebar.slider("Efficacy of vaccine", min_value=0.0 , max_value=1.0 , value=1.0 , step=0.1, key=i)
        quantity = st.sidebar.slider("Quantity of vaccines per day", min_value=1 , max_value=1000 , value=1 , step=1, key=i)
        available_vaccines['Vaccine'+str(i+1)]['cost'] = cost
        available_vaccines['Vaccine'+str(i+1)]['decay_days'] = decay
        available_vaccines['Vaccine'+str(i+1)]['efficacy'] = efficacy
        available_vaccines['Vaccine'+str(i+1)]['capacity'] = quantity
        # tc=tc+
    for vaccine in available_vaccines:
        tc=tc+ available_vaccines[vaccine]['cost']



    st.sidebar.write("------------------------------------------------------------------------------------")

    st.sidebar.write("Disease parameters")
    inf_per = st.sidebar.slider("Initial prevalence of infection", min_value=0.0 , max_value=1.0 , value=0.01 , step=0.01 , format=None , key=None )
    beta=st.sidebar.slider("Rate of infection : Susceptible->Exposed", min_value=0.0 , max_value=1.0 , value=0.3 , step=0.01 , format=None , key=None )
    mu=st.sidebar.slider("Rate of Exposed->Infected", min_value=0.0 , max_value=1.0 , value=0.7 , step=0.01 , format=None , key=None )
    gamma=st.sidebar.slider("Rate of recovery : Infected:->Recovered", min_value=0.0 , max_value=1.0 , value=0.42 , step=0.01 , format=None , key=None )
    delta=st.sidebar.slider("Rate of unimmunisation : Recovered->Susceptible", min_value=0.0 , max_value=1.0 , value=0.0 , step=0.01 , format=None , key=None )

    st.sidebar.write("------------------------------------------------------------------------------------")

    # lockdown_list=[]
    # for i in range(days):
    #     lockdown_list.append(False)
    # st.sidebar.write("Lockdown parameters")    #Average SEIR model over multiple worlds
    # num_lockdowns=st.sidebar.slider("Number of lockdowns", min_value=0 , max_value=10 , value=1 , step=1 , format=None , key=None )
    # for i in range(num_lockdowns):
    #     a,b = st.slider("Select Lockdown Range for Lockdown "+str(i+1), 1,days, (1,1), 1)
    #     for i in range(a,b+1):
    #         lockdown_list[i-1]=True
    # cum_inf, cum_ld = worlds(num_worlds,n,p,inf_per,days,beta,mu,gamma,delta,num_agents_per_step,available_vaccines)
    cum_inf= worlds(num_worlds,n,p,inf_per,days,beta,mu,gamma,delta,num_agents_per_step,available_vaccines)
    # new_sim_obj= Simulate(graph_obj,agents,transmission_prob,num_agents_per_step,available_vaccines)
    # total_vaccination_cost=new_sim_obj.enact_policy(day)
    # total_cost=

    st.write("------------------------------------------------------------------------------------")

    st.header("Cost Function")
    st.write("Goal is to minimise the cost function :")
    st.write("Cost function =  a(Cumulative Infected days) + total cost of vaccination")
    st.write(" -- 'a' refers to medical cost per infected per day")
    # st.write(" -- 'b' refers to economic loss of locking down per day")

    st.write("------------------------------------------------------------------------------------")

    a=st.slider("Medical cost per infected per day", min_value=1 , max_value=100 , value=5 , step=1 , format=None , key=None )
    # b=st.slider("Economic loss during lockdown per individual per day", min_value=1 , max_value=100 , value=1 , step=1 , format=None , key=None )

    # st.write("The Cumulative cost is "+str(a*cum_inf+b*cum_ld))
    # st.write("The Cumulative cost is "+str(a*cum_inf*cost))
    # cost=
    st.write("The total Medical cost per day is "+str(a*cum_inf))
    st.write("The total vaccination cost per day is "+str(tc))
    st.write("The Cumulative cost is "+str(a*cum_inf+cost))



main()
