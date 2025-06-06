import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'doc',
      label: 'Introduction',
      id: 'intro',
    },
    {
      type: 'category',
      label: 'Configuration',
      items: [
        'docs-configuration/configuration',
        'docs-configuration/adapters',
        'docs-configuration/drift-tests',
        'docs-configuration/usage',
      ],
    },
  ],
};

export default sidebars;
