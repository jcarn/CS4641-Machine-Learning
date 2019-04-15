package assignment4.util;

import assignment4.BasicGridWorld;
import assignment4.PokemonWorld;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.State;

public class PokeTerminalFunction implements TerminalFunction {

	int maxHealth;

	public PokeTerminalFunction(int maxH) {
		this.maxHealth = maxH;
	}

	@Override
	public boolean isTerminal(State s) {

		// get location of agent in next state
		ObjectInstance agent = s.getFirstObjectOfClass(PokemonWorld.CLASSAGENT);
		int curHealth = agent.getIntValForAttribute(PokemonWorld.ATTHEALTH);
		boolean curStatus = agent.getBooleanValForAttribute(PokemonWorld.ATTSTATUS);

		// are they captured
		if (curHealth == this.maxHealth + 1) {
			return true;
		}
		
		// are they dead/ko'd?
		if (curHealth <= 0) {
			return true;
		}

		return false;
	}

}
