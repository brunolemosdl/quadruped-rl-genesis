import Tabs, { tabs } from "./tabs.astro";
import TabsContent, { tabsContent } from "./tabs-content.astro";
import TabsList, { tabsList } from "./tabs-list.astro";
import TabsTrigger, { tabsTrigger } from "./tabs-trigger.astro";

const TabsVariants = {
  tabs,
  tabsContent,
  tabsList,
  tabsTrigger,
};

export { Tabs, TabsContent, TabsList, TabsTrigger, TabsVariants };

export default {
  Root: Tabs,
  Content: TabsContent,
  List: TabsList,
  Trigger: TabsTrigger,
};
