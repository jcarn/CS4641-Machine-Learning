package assignment4.util;

import assignment4.BasicGridWorld;
import assignment4.PokemonWorld;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;

public class PokeRewardFunction implements RewardFunction {

	int maxHealth;

	public PokeRewardFunction(int maxH) {
		this.maxHealth = maxH;
	}

	@Override
	public double reward(State s, GroundedAction a, State sprime) {

		// get location of agent in next state
		ObjectInstance agent = sprime.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
		int curHealth = agent.getIntValForAttribute(PokemonWorld.ATTHEALTH);
		boolean curStatus = agent.getBooleanValForAttribute(PokemonWorld.ATTSTATUS);

		// caught?
		if (curHealth == this.maxHealth + 1) {
			return 100.;
		}
		//ko'd the poor pokemon?
		if (curHealth <= 0) {
			return -100;
		}

		return -1;
	}

}
