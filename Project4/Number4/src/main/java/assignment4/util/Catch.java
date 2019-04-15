package assignment4.util;

import java.util.ArrayList;
import java.util.List;
import java.math.*;

import assignment4.BasicGridWorld;
import assignment4.PokemonWorld;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TransitionProbability;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.FullActionModel;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.common.SimpleAction;

public class Catch extends SimpleAction implements FullActionModel {

	//0: north; 1: south; 2:east; 3: west
	protected int maxHealth;

	public Catch(String actionName, Domain domain, int maxHealth){
		super(actionName, domain);
		this.maxHealth = maxHealth;
	}

	@Override
	protected State performActionHelper(State s, GroundedAction groundedAction) {
		//get agent and current position
		ObjectInstance agent = s.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
		int curHealth = agent.getIntValForAttribute(PokemonWorld.ATTHEALTH);
		boolean curStatus = agent.getBooleanValForAttribute(PokemonWorld.ATTSTATUS);

		//sample directon with random roll
		double r = Math.random();
		double cProb = this.catchProb(this.maxHealth, curHealth, curStatus);
		if(r < cProb) {
			//terminal state of catching is maxHealth + 1
			agent.setValue(PokemonWorld.ATTHEALTH, this.maxHealth + 1);
		}

		//return the state we just modified
		return s;
	}

	@Override
	public List<TransitionProbability> getTransitions(State s, GroundedAction groundedAction) {
		State ns;
		ObjectInstance nagent;
		//get agent and current position
		ObjectInstance agent = s.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
		int curHealth = agent.getIntValForAttribute(PokemonWorld.ATTHEALTH);
		boolean curStatus = agent.getBooleanValForAttribute(PokemonWorld.ATTSTATUS);
		double cProb = this.catchProb(maxHealth, curHealth, curStatus);

		List<TransitionProbability> tps = new ArrayList<TransitionProbability>(2);
		ns = s.copy();
		tps.add(new TransitionProbability(ns, 1.0 - cProb));
		
		ns = s.copy();
		nagent = ns.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
		nagent.setValue(PokemonWorld.ATTHEALTH, maxHealth + 1);
		tps.add(new TransitionProbability(ns, cProb));

		return tps;
	}

	protected double catchProb(int maxHealth, int curHealth, boolean curStatus) {
		double baseProb, finalProb;
		baseProb = 0.7 - Math.log((double)curHealth) / Math.log((double)maxHealth);
		if(curStatus) {
			finalProb = baseProb + .3;
		} else {
			finalProb = baseProb;
		}
		return finalProb;
	}


}

