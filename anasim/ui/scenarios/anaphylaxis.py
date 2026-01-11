from .base import Scenario, ScenarioStep

def create_anaphylaxis_scenario():
    """
    Creates a scenario for managing an anaphylactic crisis.
    
    Clinical Goals:
    1. Recognize hypotension/tachycardia (shock).
    2. Stop the triggering agent (simulated).
    3. Administer Epinephrine (drug of choice).
    4. Fluid resuscitation.
    """
    
    # Define steps
    steps = [
        # Step 1: Recognition
        ScenarioStep(
            id="ana_recognize",
            instruction="<b>Anaphylaxis suspected!</b><br>Patient has sudden hypotension and tachycardia.<br><b>Action:</b> acknowledge the crisis state.",
            requirements=lambda e: e.disturbance_active and "anaphylaxis" in str(e.disturbance_profile).lower(),
            status_success="Crisis active",
            status_fail="Start anaphylaxis in events tab"
        ),
        
        # Step 2: Epinephrine
        ScenarioStep(
            id="ana_epi",
            instruction="<b>Treat hypotension:</b><br>Administer Epinephrine bolus (10-100 mcg).<br>Target: MAP > 65 mmHg.",
            requirements=lambda e: e.get_drug_state("epi")["total_bolus"] > 0 or e.get_drug_state("epi")["rate"] > 0,
            status_success="Epinephrine given",
            status_fail="Give epinephrine"
        ),
        
        # Step 3: Fluids
        ScenarioStep(
            id="ana_fluids",
            instruction="<b>Volume resuscitation:</b><br>Administer IV fluids (at least 500 mL) to support circulation.",
            requirements=lambda e: (
                getattr(getattr(e, "hemo", None), "total_crystalloid_in_ml", 0.0) +
                getattr(getattr(e, "hemo", None), "total_blood_in_ml", 0.0)
            ) >= 500,
            status_success="Fluids administered",
            status_fail="Give fluids > 500mL"
        ),
        
        # Step 4: Stabilization
        ScenarioStep(
            id="ana_stable",
            instruction="<b>Stabilization:</b><br>Ensure MAP is stable > 65 mmHg.",
            requirements=lambda e: e.state.map > 65,
            status_success="Hemodynamics stable",
            status_fail="BP still low"
        )
    ]
    
    return Scenario(
        name="Anaphylaxis Management",
        description="Diagnose and treat a severe anaphylactic reaction.",
        steps=steps
    )
