

package assignment4;

import assignment4.util.AgentPainter;
import assignment4.util.AtLocation;
import assignment4.util.LocationPainter;
import assignment4.util.Movement;
import assignment4.util.StatusAttack;
import assignment4.util.Catch;
import assignment4.util.Attack;
import assignment4.util.WallPainter;
import burlap.oomdp.auxiliary.DomainGenerator;
import burlap.oomdp.core.Attribute;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.ObjectClass;
import burlap.oomdp.core.objects.MutableObjectInstance;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.MutableState;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.visualizer.StateRenderLayer;
import burlap.oomdp.visualizer.Visualizer;

public class PokemonWorld implements DomainGenerator {

	public static final String ATTHEALTH = "health";
	public static final String ATTSTATUS = "status";

	public static final String CLASSAGENT = "agent";
	public static final String CLASSLOCATION = "location";

	public static final String ACTIONATTACK = "attack";
	public static final String ACTIONCATCH = "catch";
	public static final String ACTIONSTATUS = "status";

	public static final String PFAT = "at";

	// ordered so first dimension is x
	public static int startHealth = 10;
	protected static int health;
	protected static boolean status;
	
	public PokemonWorld(int startH){
		this.startHealth = startH;
	}


	@Override
	public Domain generateDomain() {

		SADomain domain = new SADomain();

		Attribute hatt = new Attribute(domain, ATTHEALTH, Attribute.AttributeType.INT);
//		hatt.setLims(-3, startHealth + 1);

		Attribute satt = new Attribute(domain, ATTSTATUS, Attribute.AttributeType.BOOLEAN);

		ObjectClass agentClass = new ObjectClass(domain, CLASSAGENT);
		agentClass.addAttribute(hatt);
		agentClass.addAttribute(satt);

		new Attack(ACTIONATTACK, domain, this.startHealth);
		new Catch(ACTIONCATCH, domain, this.startHealth);
		new StatusAttack(ACTIONSTATUS, domain);

		return domain;
	}

	public static State getExampleState(Domain domain) {
		State s = new MutableState();
		ObjectInstance agent = new MutableObjectInstance(
				domain.getObjectClass(CLASSAGENT), "agent0");
		agent.setValue(ATTHEALTH, 10);
		agent.setValue(ATTSTATUS, false);


		s.addObject(agent);

		return s;
	}

//	public StateRenderLayer getStateRenderLayer() {
//		StateRenderLayer rl = new StateRenderLayer();
//		rl.addStaticPainter(new WallPainter(map));
//		rl.addObjectClassPainter(CLASSLOCATION, new LocationPainter(map));
//		rl.addObjectClassPainter(CLASSAGENT, new AgentPainter(map));
//
//		return rl;
//	}

//	public Visualizer getVisualizer() {
//		return new Visualizer(this.getStateRenderLayer());
//	}

	public int getStartHealth() {
		return this.startHealth;
	}

	public void setStartHealth(int sH) {
		this.startHealth = sH;
	}



}
