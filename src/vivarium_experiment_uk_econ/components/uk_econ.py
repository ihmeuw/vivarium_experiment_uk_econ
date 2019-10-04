"""Adapted from vivarium tutorial https://vivarium.readthedocs.io/en/latest/tutorials/disease_model.html
"""

import pandas as pd
import numpy as np
import scipy.stats, scipy.interpolate

from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.event import Event

from vivarium_experiment_uk_econ.external_data.data import LOCAL_DATA_DIR

class BasePopulation:
    """Generates a base population with a uniform distribution of age and sex.

    Attributes
    ----------
    configuration_defaults :
        A set of default configuration values for this component. These can be
        overwritten in the simulation model specification or by providing
        override values when constructing an interactive simulation.
    """

    configuration_defaults = {
        'population': {
            # The range of ages to be generated in the initial population
            'age_start': 0,
            'age_end': 100,
            # Note: There is also a 'population_size' key.
        },
    }

    def __init__(self):
        self.name = 'base_population'

    def setup(self, builder: Builder):
        """Performs this component's simulation setup.

        The ``setup`` method is automatically called by the simulation
        framework. The framework passes in a ``builder`` object which
        provides access to a variety of framework subsystems and metadata.

        Parameters
        ----------
        builder :
            Access to simulation tools and subsystems.
        """
        self.config = builder.configuration

        self.with_common_random_numbers = bool(self.config.randomness.key_columns)
        self.register = builder.randomness.register_simulants
        if (self.with_common_random_numbers
                and not ['entrance_time', 'age'] == self.config.randomness.key_columns):
            raise ValueError("If running with CRN, you must specify ['entrance_time', 'age'] as"
                             "the randomness key columns.")

        self.age_randomness = builder.randomness.get_stream('age_initialization',
                                                            for_initialization=self.with_common_random_numbers)
        self.sex_randomness = builder.randomness.get_stream('sex_initialization')

        columns_created = ['age', 'sex', 'alive', 'entrance_time']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created)

        self.population_view = builder.population.get_view(columns_created)

        builder.event.register_listener('time_step', self.age_simulants)

    def on_initialize_simulants(self, pop_data: SimulantData):
        """Called by the simulation whenever new simulants are added.

        This component is responsible for creating and filling four columns
        in the population state table:

        'age' :
            The age of the simulant in fractional years.
        'sex' :
            The sex of the simulant. One of {'Male', 'Female'}
        'alive' :
            Whether or not the simulant is alive. One of {'alive', 'dead'}
        'entrance_time' :
            The time that the simulant entered the simulation. The 'birthday'
            for simulants that enter as newborns. A `pandas.Timestamp`.

        Parameters
        ----------
        pop_data :
            A record containing the index of the new simulants, the
            start of the time step the simulants are added on, the width
            of the time step, and the age boundaries for the simulants to
            generate.

        """

        age_start = self.config.population.age_start
        age_end = self.config.population.age_end
        if age_start == age_end:
            age_window = pop_data.creation_window / pd.Timedelta(days=365)
        else:
            age_window = age_end - age_start

        age_draw = self.age_randomness.get_draw(pop_data.index)
        age = age_start + age_draw * age_window

        if self.with_common_random_numbers:
            population = pd.DataFrame({'entrance_time': pop_data.creation_time,
                                       'age': age.values}, index=pop_data.index)
            self.register(population)
            population['sex'] = self.sex_randomness.choice(pop_data.index, ['Male', 'Female'])
            population['alive'] = 'alive'
        else:
            population = pd.DataFrame(
                {'age': age.values,
                 'sex': self.sex_randomness.choice(pop_data.index, ['Male', 'Female']),
                 'alive': pd.Series('alive', index=pop_data.index),
                 'entrance_time': pop_data.creation_time},
                index=pop_data.index)

        self.population_view.update(population)

    def age_simulants(self, event: Event):
        """Updates simulant age on every time step.

        Parameters
        ----------
        event :
            An event object emitted by the simulation containing an index
            representing the simulants affected by the event and timing
            information.
        """
        population = self.population_view.get(event.index, query="alive == 'alive'")
        population['age'] += event.step_size / pd.Timedelta(days=365)
        self.population_view.update(population)



class Mortality:
    """Introduces death into the simulation.

    Attributes
    ----------
    configuration_defaults :
        A set of default configuration values for this component. These can be
        overwritten in the simulation model specification or by providing
        override values when constructing an interactive simulation.
    """

    configuration_defaults = {
        'mortality': {
            'mortality_rate': 0.01,
        }
    }

    def __init__(self):
        self.name = 'mortality'

    def setup(self, builder: Builder):
        """Performs this component's simulation setup.

        The ``setup`` method is automatically called by the simulation
        framework. The framework passes in a ``builder`` object which
        provides access to a variety of framework subsystems and metadata.

        Parameters
        ----------
        builder :
            Access to simulation tools and subsystems.
        """
        self.config = builder.configuration.mortality
        self.population_view = builder.population.get_view(['alive'], query="alive == 'alive'")
        self.randomness = builder.randomness.get_stream('mortality')

        self.mortality_rate = builder.value.register_rate_producer('mortality_rate', source=self.base_mortality_rate)

        builder.event.register_listener('time_step', self.determine_deaths)

    def base_mortality_rate(self, index: pd.Index) -> pd.Series:
        """Computes the base mortality rate for every individual.

        Parameters
        ----------
        index :
            A representation of the simulants to compute the base mortality
            rate for.

        Returns
        -------
            The base mortality rate for all simulants in the index.
        """
        return pd.Series(self.config.mortality_rate, index=index)

    def determine_deaths(self, event: Event):
        """Determines who dies each time step.

        Parameters
        ----------
        event :
            An event object emitted by the simulation containing an index
            representing the simulants affected by the event and timing
            information.
        """
        effective_rate = self.mortality_rate(event.index)
        effective_probability = 1 - np.exp(-effective_rate)
        draw = self.randomness.get_draw(event.index)
        affected_simulants = draw < effective_probability
        self.population_view.update(pd.Series('dead', index=event.index[affected_simulants]))


class Income:
    """Adds income attributes as pipelines (income = before-tax income;
    after_tax_income and tax_amount are what you would guess, and
    net_income is a pipeline which will be modified to include after
    tax income plus share of non-health tax benefits), based on
    income_propensity column that is added to population table
    (initialized on creation)

    """

    def __init__(self):
        self.name = 'income'

    def setup(self, builder: Builder):
        self.config = builder.configuration
        self.income_randomness = builder.randomness.get_stream('income_initialization')

        columns_created = ['income_propensity', 'utility', 'taxes']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created)
        self.population_view = builder.population.get_view(columns_created, query="alive == 'alive'")

        self.clock = builder.time.clock()

        self.income = builder.value.register_value_producer('income', source=self.get_income)
        self.after_tax_income = builder.value.register_value_producer('after_tax_income', source=self.get_after_tax_income)
        self.tax_amount = builder.value.register_value_producer('tax_amount', source=self.get_tax_amount)
        self.net_income = builder.value.register_value_producer('net_income', source=self.get_net_income)

        # load data on income, make interpolaters
        self.income_func = {}
        for when in ['before_tax', 'after_tax']:
            fname = LOCAL_DATA_DIR.joinpath('income_data.xlsx')
            df = pd.read_excel(fname, sheetname=f'{when}_income', index_col='percentile')
            self.income_func[when] = \
                        scipy.interpolate.interp2d(df.index, df.columns, df.values.T)

        builder.event.register_listener('time_step', self.accrue_values)

    def on_initialize_simulants(self, pop_data: SimulantData):
        income_propensity = self.income_randomness.get_draw(pop_data.index)
        population = pd.DataFrame({
            'income_propensity': income_propensity,
            })
        population['utility'] = 0.0
        population['taxes'] = 0.0
        self.population_view.update(population)

    def get_income(self, index: pd.Index) -> pd.Series:
        year = self.clock().year
        pop = self.population_view.get(index)
        return self.income_func['before_tax'](100*pop.income_propensity, year)

    def get_after_tax_income(self, index: pd.Index) -> pd.Series:
        year = self.clock().year
        pop = self.population_view.get(index)
        return self.income_func['after_tax'](100*pop.income_propensity, year)

    def get_tax_amount(self, index: pd.Index) -> pd.Series:
        return self.income(index) - self.after_tax_income(index)

    def get_net_income(self, index: pd.Index) -> pd.Series:
        return self.after_tax_income(index)

    def accrue_values(self, event: Event):
        """increment taxes and utility for each individual

        Parameters
        ----------
        event :
            An event object emitted by the simulation containing an index
            representing the simulants affected by the event and timing
            information.
        """
        pop = self.population_view.get(event.index)
        step_size = event.step_size / pd.Timedelta('365.25 days')

        # Our utility follows utility function from Atkinson, Measurement
        # of Inequality (p. 251)
        # U(y) = A + B y**(1-eps)/(1-eps) if eps \neq 1
        #      = log(y)                   if eps = 1

        # Atkinson varied eps from 1 to 2.5, but I don't know what A
        # or B should be.  How about A = 0 and B = 1?

        eps = .5
        if eps == 1:
            utility_rate = np.log(self.net_income(event.index))
        elif eps == 0:
            utility_rate = self.net_income(event.index)
        else:
            utility_rate = (self.net_income(event.index))**(1-eps) / (1-eps)
            
        pop.utility += utility_rate * step_size

        taxes = self.tax_amount(event.index)
        pop.taxes += taxes * step_size

        self.population_view.update(pop)


class Taxes:
    """Adds tax burden and benefits, initialized on creation

    Attributes
    ----------
    configuration_defaults :
        A set of default configuration values for this component. These can be
        overwritten in the simulation model specification or by providing
        override values when constructing an interactive simulation.
    """

    configuration_defaults = {
        'taxes' : {
            'non_health_fraction': .9,  # TODO: find this fraction empirically from GHE estimates
            'health_benefit': 0.0001,
        }
    }

    def __init__(self):
        self.name = 'taxes'

    def setup(self, builder: Builder):
        self.config = builder.configuration

        self.population_view = builder.population.get_view(['income_propensity'], query="alive == 'alive'")

        self.individual_taxes = builder.value.get_value('tax_amount')
        self.total_taxes = builder.value.register_value_producer('total_taxes', source=self.get_total_taxes)
        builder.value.register_value_modifier('net_income', modifier=self.increase_net_income)
        builder.value.register_value_modifier('mortality_rate', modifier=self.reduce_mortality)

    def get_total_taxes(self, index: pd.Index) -> float:
        return np.sum(self.individual_taxes(index))

    def increase_net_income(self, index: pd.Index, net_income: pd.Series) -> pd.Series:
        return (net_income 
                + self.config.taxes.non_health_fraction * self.total_taxes(index) / len(index))

    def reduce_mortality(self, index: pd.Index, mortality_rate: pd.Series) -> pd.Series:
        return mortality_rate - self.config.taxes.health_benefit


class UtilityObserver():
    def __init__(self):
        self.name = 'utility_observer'

    def setup(self, builder):
        self.config = builder.configuration
        self.total_taxes = builder.value.get_value('total_taxes')

        self.population_view = builder.population.get_view(['alive', 'utility', 'taxes'])
        self.gross_income = builder.value.get_value('income')
        self.net_income = builder.value.get_value('net_income')
        builder.value.register_value_modifier('metrics', modifier=self.metrics)

    def metrics(self, index, metrics):
        pop = self.population_view.get(index)
        metrics.update({'death_count': np.sum(pop.alive == 'dead'),
                        'utility': np.sum(pop.utility),
                        'taxes_spent_on_health': (1-self.config.taxes.non_health_fraction) * np.sum(pop.taxes),
                        'gdp_pc': self.gross_income(index).mean(),
                    })

        return metrics

