import 'fake-indexeddb/auto';
import { IDBFactory } from 'fake-indexeddb';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { appendMemoryEvent, closeMemoryDb, deduplicateActivity, loadMemory, MAX_EVENTS_PER_CONVERSATION, selectMemoryConversation } from './memory';
import type { Activity } from './types';

describe('conversation memory', () => {
  beforeEach(() => {
    closeMemoryDb();
    globalThis.indexedDB = new IDBFactory();
  });

  afterEach(() => {
    closeMemoryDb();
    vi.restoreAllMocks();
  });

  it('removes only exact duplicate events from earlier Strict Mode logging', () => {
    const event: Activity = {
      id: 'first', source: 'person', message: 'What should I pack?', at: '3:55 PM', createdAt: '2026-07-25T19:55:00.000Z'
    };
    const sameEvent = { ...event, id: 'duplicate' };
    const intentionalRepeat = { ...event, id: 'later', createdAt: '2026-07-25T19:56:00.000Z' };

    expect(deduplicateActivity([event, sameEvent, intentionalRepeat]).map((item) => item.id))
      .toEqual(['first', 'later']);
  });

  it('keeps only the newest events for each conversation', async () => {
    const { activeConversation } = await loadMemory();
    const total = MAX_EVENTS_PER_CONVERSATION + 5;
    for (let index = 0; index < total; index += 1) {
      await appendMemoryEvent(activeConversation.id, {
        id: `event-${index}`, source: 'person', message: `message ${index}`, at: '', order: index,
        createdAt: new Date(Date.UTC(2026, 0, 1, 0, 0, index)).toISOString()
      });
    }

    const { activity } = await selectMemoryConversation(activeConversation.id);
    expect(activity).toHaveLength(MAX_EVENTS_PER_CONVERSATION);
    expect(activity[0].id).toBe(`event-${total - 1}`);
    expect(activity.at(-1)?.id).toBe(`event-${total - MAX_EVENTS_PER_CONVERSATION}`);
  });

  it('reuses one database connection across operations', async () => {
    const open = vi.spyOn(globalThis.indexedDB, 'open');
    const { activeConversation } = await loadMemory();
    await appendMemoryEvent(activeConversation.id, { id: 'one', source: 'person', message: 'hello', at: '', order: 1 });
    await selectMemoryConversation(activeConversation.id);
    expect(open).toHaveBeenCalledTimes(1);
  });
});
