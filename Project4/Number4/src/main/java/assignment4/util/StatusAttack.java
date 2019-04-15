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

public class StatusAttack extends SimpleAction implements FullActionModel {

	//0: north; 1: south; 2:east; 3: west
	protected double statusProbs;

	public StatusAttack(String actionName, Domain domain){
		super(actionName, domain);
		statusProbs = .9;
	}

	@Override
	protected State performActionHelper(State s, GroundedAction groundedAction) {
		//get agent and current position
		ObjectInstance agent = s.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
		int curX = agent.getIntValForAttribute(PokemonWorld.ATTHEALTH);
		boolean curStatus = agent.getBooleanValForAttribute(PokemonWorld.ATTSTATUS);

		boolean newStatus;
		//sample directon with random roll
		double r = Math.random();
		
		if(r < statusProbs || curStatus){
			newStatus = true;
		} else {
			newStatus = false;
		}

		//set the new position
		agent.setValue(PokemonWorld.ATTSTATUS, newStatus);

		//return the state we just modified
		return s;
	}

	@Override
	public List<TransitionProbability> getTransitions(State s, GroundedAction groundedAction) {
		//get agent and current position
		ObjectInstance agent = s.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
		boolean curStatus = agent.getBooleanValForAttribute(PokemonWorld.ATTSTATUS);
		ObjectInstance nagent;
		
		List<TransitionProbability> tps = new ArrayList<TransitionProbability>(2);
		TransitionProbability noChangeTransition = null;
		State ns = s.copy();
		if(curStatus) {
			nagent = ns.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
			nagent.setValue(PokemonWorld.ATTSTATUS, false);
			tps.add(new TransitionProbability(ns, 0.0));
			tps.add(new TransitionProbability(s, 1.0));
		} else {
			ns = s.copy();
			nagent = ns.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
			nagent.setValue(PokemonWorld.ATTSTATUS, false);
			tps.add(new TransitionProbability(ns, 1.0 - statusProbs));
			ns = s.copy();
			nagent = ns.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
			nagent.setValue(PokemonWorld.ATTSTATUS, false);
			tps.add(new TransitionProbability(ns, statusProbs));
		}
		
		return tps;
	}
}

