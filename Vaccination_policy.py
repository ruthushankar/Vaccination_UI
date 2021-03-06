import random
import copy
from functools import partial


class Result():
    def __init__(self,vaccine_name,agent,result,time_step,efficacy,decay_days,vaccine_cost):
        self.vaccine_name=vaccine_name
        self.agent=agent
        self.result=result
        self.time_stamp=time_step
        self.protection= decay_days
        self.vaccine_cost=vaccine_cost


class Vaccine_type():
    def __init__(self,name,cost,decay,efficacy):

        self.vaccine_name=name
        self.vaccine_cost=cost
        self.decay_days=decay
        self.efficacy=efficacy
        self.total=0



    def vaccinate(self,agent,time_step):

        # vaccinate agents

        result=self.inject_agent(agent)
        result_obj= Result(self.vaccine_name,agent,result,time_step,self.efficacy,self.decay_days,self.vaccine_cost)



        return result_obj


    def inject_agent(self,agent):

        if (random.random()<self.efficacy):
            agent.inject=True

            return 'Successful'
        else:
            return "Unsuccessful"




class Vaccination_policy():
    def __init__(self,agents_per_step_fn=None):
        super().__init__()

        self.policy_type='Vaccination'
        self.available_vaccines={}
        self.vaccines=[]
        self.statistics={}
        self.statistics_total={}
        self.statistics_total['Total Vaccination']=[]
        self.statistics_total['Total Successful']=[]
        self.statistics_total['Total Unsuccessful']=[]
        assert callable(agents_per_step_fn)
        self.agents_per_step_fn = agents_per_step_fn
        self.total_cost=0


    def enact_policy(self,time_step,agents):

        # self.total_cost1=self.newday(time_step)
        self.newday(time_step)
        self.set_protection(agents)
        fn=self.full_random_vaccines()
        fn(agents,time_step)
        f_cost=self.populate_results()
        self.restrict_agents(agents)
        self.get_stats()

        # return self.total_cost1
        return f_cost




    def newday(self,time_step):

        self.vaccines=[]
        self.results=[]
        self.num_agents_to_vaccinate = self.agents_per_step_fn(time_step)
        self.cumulative_cost=0

        for name in self.available_vaccines.keys():

            for i in range(int(self.available_vaccines[name]['number'])):
                name,cost,decay,efficacy=self.available_vaccines[name]['parameters']
                vaccine_obj=Vaccine_type(name,cost,decay,efficacy)
                # self.total_cost+=(vaccine_obj.vaccine_cost* self.available_vaccines[name]['number'])
                self.total_cost=(vaccine_obj.vaccine_cost* self.available_vaccines[name]['number'])
                self.vaccines.append(vaccine_obj)
            self.cumulative_cost+=self.total_cost

        # return self.total_cost
        return self.cumulative_cost





    def full_random_vaccines(self,parameter=None, value_list=[]):


        assert isinstance(value_list,list)
        return partial(self.random_vaccination,parameter, value_list)

    def random_vaccination(self,parameter,value_list,agents,time_step):
        agents_copy = copy.copy(list(agents))
        random.shuffle(agents_copy)
        curr_agents_to_vaccinate= self.num_agents_to_vaccinate



        for agent in agents_copy:
            if (curr_agents_to_vaccinate<=0):
                break

            # if (agent.get_policy_state('Vaccination') is None and len(self.vaccines)):
            if agent.vaccination_state==None and len(self.vaccines):
                if parameter is None or agent.info[parameter] in value_list:

                    current_vaccine= random.choice(self.vaccines)
                    result=current_vaccine.vaccinate(agent,time_step)
                    self.results.append(result)
                    self.vaccines.remove(current_vaccine)
                    curr_agents_to_vaccinate-=1


    def set_protection(self,agents):
        for agent in agents:
            # history= self.get_agent_policy_history(agent)
            history= agent.vaccination_hist
            # dict of result objects
            if len(history)==0:
                continue
            else:

                history[-1].protection-=1


    def populate_results(self):
        self.costy1=0
        for result_obj in self.results:
            agent= result_obj.agent
            costy=result_obj.vaccine_cost
            self.costy1+=costy
            # self.update_agent_policy_history(agent,result_obj)
            agent.vaccination_hist.append(result_obj)
            # self.update_agent_policy_state(agent,result_obj.result)
            agent.vaccination_state=result_obj.result

        return self.costy1


    def restrict_agents(self,agents):
        for agent in agents:
            # history=self.get_agent_policy_history(agent)

            history= agent.vaccination_hist

            if (len(history)!=0):
                if(history[-1].result=="Successful"):
                    if(history[-1].protection>=1):
                        agent.restrict=True
                        # agent.restrict_recieve_infection()


    def get_stats(self):
        self.statistics_total['Total Vaccination'].append(0)
        self.statistics_total['Total Successful'].append(0)
        self.statistics_total['Total Unsuccessful'].append(0)
        for name in self.available_vaccines.keys():
            self.statistics[name]['Total Vaccination'].append(0)
            self.statistics[name]['Total Successful'].append(0)
            self.statistics[name]['Total Unsuccessful'].append(0)

        for result_obj in self.results:
            self.statistics_total['Total Vaccination'][-1]+=1
            name=result_obj.vaccine_name
            self.statistics[name]['Total Vaccination'][-1]+=1
            result=result_obj.result
            if result=="Successful":
                self.statistics[name]['Total Successful'][-1]+=1
                self.statistics_total['Total Successful'][-1]+=1
            elif result=="Unsuccessful":
                self.statistics[name]['Total Unsuccessful'][-1]+=1
                self.statistics_total['Total Unsuccessful'][-1]+=1

    def add_vaccination(self,name,cost,decay,efficacy,num):

        if name in self.available_vaccines.keys():
            if [name,cost,decay,efficacy]==self.available_vaccines[name]['parameters']:
                self.available_vaccines[name]['number']+=num
                self.statistics[name]={'Total Vaccination':[],'Total Successful':[],'Total Unsuccessful':[]}
            else:
                print("Error! Vaccine name with different parameter exists")


        else:
            self.available_vaccines[name]={'parameters':[name,cost,decay,efficacy],'number':num}
            self.statistics[name]={'Total Vaccination':[],'Total Successful':[],'Total Unsuccessful':[]}
