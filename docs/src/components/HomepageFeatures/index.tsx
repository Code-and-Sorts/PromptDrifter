import type {ReactNode} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Catch Drift Early',
    Svg: require('@site/static/img/undraw_ai-code-generation.svg').default,
    description: (
      <>
        Detect and be alerted when your LLM responses change unexpectedly before they cause issues in production.
        Stop prompt drift at the source.
      </>
    ),
  },
  {
    title: 'Model Comparison and Analysis',
    Svg: require('@site/static/img/undraw_algorithm-execution.svg').default,
    description: (
      <>
        Compare responses across different LLM providers, temperature settings, and prompt variations side-by-side.
        Isolate and measure the impact of subtle changes to identify the optimal configuration for your use case.
      </>
    ),
  },
  {
    title: 'Multiple LLM Support',
    Svg: require('@site/static/img/undraw_ai-agent.svg').default,
    description: (
      <>
        Works with OpenAI, Anthropic, Google, and many other LLM providers
        through an extensible adapter system.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className={styles.highlightCard}>
        <div className={styles.svgContainer}>
          <Svg className={styles.featureSvg} role="img" />
        </div>
        <div className="text--center padding-horiz--md">
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.highlights}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}