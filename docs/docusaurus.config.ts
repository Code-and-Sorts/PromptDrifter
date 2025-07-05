import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'PromptDrifter',
  tagline: 'One-command CI guardrail that catches prompt drift and fails the build when your LLM answers change',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://code-and-sorts.github.io',
  baseUrl: '/PromptDrifter/',

  organizationName: 'Code-and-Sorts',
  projectName: 'PromptDrifter',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/Code-and-Sorts/PromptDrifter/tree/main/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'PromptDrifter',
      logo: {
        alt: 'PromptDrifter Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://github.com/Code-and-Sorts/PromptDrifter',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/intro',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/Code-and-Sorts/PromptDrifter/discussions',
            },
            {
              label: 'Issues',
              href: 'https://github.com/Code-and-Sorts/PromptDrifter/issues',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/Code-and-Sorts/PromptDrifter',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Code and Sorts. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
    announcementBar: {
      id: 'support_us_3',
      content:
        '🌟 If you like PromptDrifter, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/Code-and-Sorts/PromptDrifter">GitHub</a>! 🌟',
      backgroundColor: '#4bcbf1',
      textColor: '#000000',
      isCloseable: true,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
