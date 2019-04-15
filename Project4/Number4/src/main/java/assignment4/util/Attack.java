package assignment4.util;

import java.util.ArrayList;
import java.util.List;

import assignment4.BasicGridWorld;
import assignment4.PokemonWorld;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TransitionProbability;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.FullActionModel;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.common.SimpleAction;

public class Attack extends SimpleAction implements FullActionModel {

	//0: north; 1: south; 2:east; 3: west
	protected double [] damageProbs = new double[4];
	protected int startHealth;

	public Attack(String actionName, Domain domain, int startHealth){
		super(actionName, domain);
		for(int i = 0; i < 4; i++){
			damageProbs[0] = .1;
			damageProbs[1] = .3;
			damageProbs[2] = .4;
			damageProbs[3] = .2;
		}
		this.startHealth = startHealth;
	}

	@Override
	protected State performActionHelper(State s, GroundedAction groundedAction) {
		//get agent and current position
		ObjectInstance agent = s.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
		int curHealth = agent.getIntValForAttribute(PokemonWorld.ATTHEALTH);
		boolean curStatus = agent.getBooleanValForAttribute(PokemonWorld.ATTSTATUS);

		//sample directon with random roll
		double r = Math.random();
		double sumProb = 0.;
		int damage = 0;
		for(int i = 0; i < this.damageProbs.length; i++){
			sumProb += this.damageProbs[i];
			if(r < sumProb){
				damage = i;
				break; //found direction
			}
		}

		//get resulting position
		int newHealth = curHealth - damage;
		if(newHealth < 0) {
			newHealth = 0;
		}

		//set the new position
		agent.setValue(PokemonWorld.ATTHEALTH, newHealth);

		//return the state we just modified
		return s;
	}

	@Override
	public List<TransitionProbability> getTransitions(State s, GroundedAction groundedAction) {
		//get agent and current position
		ObjectInstance agent = s.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
		int curHealth = agent.getIntValForAttribute(PokemonWorld.ATTHEALTH);
		boolean curStatus = agent.getBooleanValForAttribute(PokemonWorld.ATTSTATUS);

		List<TransitionProbability> tps = new ArrayList<TransitionProbability>(4);
		TransitionProbability noChangeTransition = null;
		for(int i = 0; i < this.damageProbs.length; i++){
			int newHealth = curHealth - i;
			if(newHealth < 0) {
				newHealth = 0;
			}
			//new possible outcome
			State ns = s.copy();
			ObjectInstance nagent = ns.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
			nagent.setValue(PokemonWorld.ATTHEALTH, newHealth);
			nagent.setValue(PokemonWorld.ATTSTATUS, curStatus);

			//create transition probability object and add to our list of outcomes
			tps.add(new TransitionProbability(ns, this.damageProbs[i]));
		}

		return tps;
	}


}

